"""
BanglaSQL — Phase 3: Training Script
Seq2Seq fine-tuning for Bangla → SQL generation.

Models: csebuetnlp/banglat5 (primary) | google/mt5-small (fallback)

== Quick Start on Colab ==
1. Clone the repo (or upload the project folder)
2. !pip install -r requirements_colab.txt -q
3. !python create_database.py
4. !python build_dataset.py
5. !python preprocess_check.py
6. !python train.py
7. !python evaluate.py

== Fallback trigger ==
If BanglaT5 does not reach >20% execution accuracy on the dev set after 10
epochs, set "model_name" to "google/mt5-small" in data/train_config.json and
retrain.

== Notes on two non-obvious choices ==
- tie_word_embeddings is left alone. T5 physically shares one tensor between
  embed_tokens and lm_head; flipping the config flag without actually untying
  the weights makes safetensors save a checkpoint with those tensors missing,
  and they are randomly re-initialised on reload — a silently corrupted model.
  The HF warning this would silence is cosmetic.
- fp16 stays off. T5-family activations overflow in standard FP16 and produce
  NaN losses.
"""

import json
import os
import random
import logging
import inspect

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from common import format_input, load_config, load_split

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TRAIN_CONFIG      = load_config()
MODEL_NAME        = TRAIN_CONFIG["model_name"]
MAX_INPUT_LENGTH  = int(TRAIN_CONFIG["max_input_length"])
MAX_TARGET_LENGTH = int(TRAIN_CONFIG["max_target_length"])
SCHEMA_STRING     = TRAIN_CONFIG["schema_string"]

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Sized for ~640 training pairs: at batch 8 that is ~80 optimizer steps/epoch, so
# 40 epochs gives ~3.2k steps — enough for BanglaT5 to converge on this task.
# Gradient accumulation is off; with a dataset this small, more frequent updates
# converge faster than a larger effective batch.
BATCH_SIZE       = 8
EVAL_BATCH_SIZE  = 16
GRAD_ACCUM_STEPS = 1
LEARNING_RATE    = 3e-4
NUM_EPOCHS       = 40
WEIGHT_DECAY     = 0.01
SAVE_TOTAL_LIMIT = 2
LABEL_SMOOTHING  = 0.0    # off: exact SQL token generation needs sharp probabilities

EARLY_STOPPING_PATIENCE      = 6
MIN_EPOCHS_BEFORE_EARLY_STOP = 8

# Eval-time beams match the beams used at inference so eval_exact_match predicts
# real-world behaviour.
EVAL_NUM_BEAMS = 4


class BanglaSQLDataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, schema: str):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.schema    = schema

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        model_inputs = self.tokenizer(
            format_input(pair["bangla_question"], self.schema),
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            text_target=pair["sql_query"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class WarmupEarlyStoppingCallback(EarlyStoppingCallback):
    """EarlyStoppingCallback that ignores evaluations before `min_epochs`.

    exact_match reads 0.0 for the first several epochs while the model is still
    learning SQL syntax, so plain patience would burn through and stop training
    long before convergence.
    """

    def __init__(self, early_stopping_patience=4, early_stopping_threshold=0.0, min_epochs=0):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        self.min_epochs = min_epochs

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.epoch is not None and state.epoch < self.min_epochs:
            logger.info(
                f"[early-stopping] epoch {state.epoch:.2f} < warm-up "
                f"({self.min_epochs}) — skipping check."
            )
            return
        super().on_evaluate(args, state, control, metrics, **kwargs)


def compute_metrics(eval_preds, tokenizer):
    """Exact match on whitespace-normalized SQL strings."""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds  = np.asarray(preds)
    labels = np.asarray(labels)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    preds  = np.where((preds != -100) & (preds >= 0), preds, pad_id)
    labels = np.where((labels != -100) & (labels >= 0), labels, pad_id)

    decoded_preds  = [" ".join(p.split()) for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    decoded_labels = [" ".join(l.split()) for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

    total = len(decoded_preds)
    if total == 0:
        return {"exact_match": 0.0}
    matches = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    return {"exact_match": round(matches / total, 4)}


def main():
    logger.info("=" * 60)
    logger.info("BanglaSQL — Phase 3: Training")
    logger.info("=" * 60)
    logger.info(f"Model      : {MODEL_NAME}")
    logger.info(f"Max in/out : {MAX_INPUT_LENGTH}/{MAX_TARGET_LENGTH}")
    logger.info(f"Batch      : {BATCH_SIZE} x{GRAD_ACCUM_STEPS} accum")
    logger.info(f"Epochs     : {NUM_EPOCHS}   LR: {LEARNING_RATE}")
    logger.info(f"Early stop : patience={EARLY_STOPPING_PATIENCE}, warm-up={MIN_EPOCHS_BEFORE_EARLY_STOP}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device     : {device}")
    if device == "cpu":
        logger.warning("No GPU detected — training will be slow. Use Colab/Kaggle T4 GPU.")
    else:
        torch.cuda.empty_cache()
        logger.info(f"GPU        : {torch.cuda.get_device_name(0)}")

    logger.info(f"\nLoading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    logger.info("\nLoading datasets...")
    train_pairs = load_split("train")
    dev_pairs   = load_split("dev")
    logger.info(f"  Train: {len(train_pairs)} pairs")
    logger.info(f"  Dev  : {len(dev_pairs)} pairs")

    train_dataset = BanglaSQLDataset(train_pairs, tokenizer, SCHEMA_STRING)
    dev_dataset   = BanglaSQLDataset(dev_pairs, tokenizer, SCHEMA_STRING)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    args_sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": CHECKPOINTS,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_exact_match",
        "greater_is_better": True,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "predict_with_generate": True,
        "generation_max_length": MAX_TARGET_LENGTH,
        "generation_num_beams": EVAL_NUM_BEAMS,
        "logging_dir": LOGS_DIR,
        "logging_steps": 20,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "fp16": False,
        "label_smoothing_factor": LABEL_SMOOTHING,
    }

    if "eval_strategy" in args_sig:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in args_sig:
        training_kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in args_sig:
        training_kwargs["save_strategy"] = "epoch"

    training_args = Seq2SeqTrainingArguments(
        **{k: v for k, v in training_kwargs.items() if k in args_sig}
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": data_collator,
        "compute_metrics": lambda p: compute_metrics(p, tokenizer),
        "callbacks": [
            WarmupEarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                min_epochs=MIN_EPOCHS_BEFORE_EARLY_STOP,
            )
        ],
    }
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    logger.info("\nStarting training...")
    trainer.train()

    best_model_dir = os.path.join(CHECKPOINTS, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info(f"Best model saved to: {best_model_dir}")

    # Per-epoch loss/metric history for the report's training curves.
    history_path = os.path.join(LOGS_DIR, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=float)
    logger.info(f"Training history saved to: {history_path}")

    logger.info("\nFinal dev set evaluation...")
    dev_metrics = trainer.evaluate()
    with open(os.path.join(LOGS_DIR, "dev_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2, default=float)

    best_em = dev_metrics.get("eval_exact_match", 0)
    logger.info("\n" + "=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Dev exact_match : {best_em}")
    logger.info(f"  Best model at   : {best_model_dir}")
    logger.info("=" * 60)
    logger.info("Next: python evaluate.py   (execution accuracy on the test split)")

    if isinstance(best_em, (int, float)) and best_em < 0.20:
        logger.warning(
            "\n[FALLBACK TRIGGER] Dev exact_match < 20%. Consider setting "
            "\"model_name\" to \"google/mt5-small\" in data/train_config.json and retraining."
        )


if __name__ == "__main__":
    main()
