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
If BanglaT5 does not reach >20% dev execution accuracy after 10 epochs, set
"model_name" to "google/mt5-small" in data/train_config.json and retrain.

== Notes on non-obvious choices ==
- tie_word_embeddings is left alone. T5 physically shares one tensor between
  embed_tokens and lm_head; flipping the config flag without actually untying
  the weights makes safetensors save a checkpoint with those tensors missing,
  and they are randomly re-initialised on reload — a silently corrupted model.
- fp16 stays off. T5-family activations overflow in standard FP16 and produce
  NaN losses.
- The best checkpoint is chosen on dev execution accuracy, not exact match.
  Exact match rejects semantically identical SQL (other alias or column order),
  so the "best" epoch was partly decided by surface form.
- Dev evaluation during training decodes greedily. Beam search there cost about
  a third of every epoch; evaluate.py applies beams plus execution-guided
  decoding to the final checkpoint.
"""

import json
import os
import random
import logging
import inspect
import warnings

import huggingface_hub
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from common import (
    DB_PATH, format_input, load_config, load_split, normalize_sql,
    open_readonly_db, results_match, run_sql, save_model_config,
)
from table_head import TABLES, TableHead, encode_tables, pool, save_head

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# No timestamp prefix: the training log is read as a sequence of epochs, not of clock
# times, and the prefix pushed the numbers that matter off to the right.
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Keep the log to the two things worth reading: the periodic loss dict and the metrics
# dict at the end of each epoch. Everything muted below is either a repainting progress
# bar, an HTTP request the hub made, or a banner restating settings already printed.
for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub", "datasets"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
transformers.utils.logging.set_verbosity_warning()
transformers.utils.logging.disable_progress_bar()
huggingface_hub.utils.disable_progress_bars()
# Raised once per epoch on a CPU-only machine; irrelevant on the GPU the run uses.
warnings.filterwarnings("ignore", message=".*pin_memory.*")

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

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Runs peak around epoch 7-9, so 25 epochs with patience 5 covers the window.
# LR is 2e-4 rather than 3e-4: at 3e-4 run 2 blew up at epoch 10 (dev loss
# 0.22 -> 0.80, execution accuracy 0.54 -> 0.07) before recovering, which is a
# wasted epoch and risks early stopping firing on the dip.
BATCH_SIZE       = 8
EVAL_BATCH_SIZE  = 32
GRAD_ACCUM_STEPS = 1
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 25
WEIGHT_DECAY     = 0.01
SAVE_TOTAL_LIMIT = 2
LABEL_SMOOTHING  = 0.0

# Patience is deliberately loose. Dev is 261 examples from 29 templates and dev
# execution accuracy swings by ~0.10 between adjacent epochs, so a normal dip looks
# like a plateau. At patience=5 run 4 stopped at epoch 11 with its best at epoch 6
# (dev 0.517), while run 5 — same setting, luckier curve — ran to 21 and peaked at
# epoch 16 (dev 0.605), worth about 20 points of test accuracy. Stopping early costs
# far more here than a few extra epochs do.
EARLY_STOPPING_PATIENCE      = 10
MIN_EPOCHS_BEFORE_EARLY_STOP = 6

# One loss line per this many optimizer steps. At ~190 steps an epoch that is about
# nine lines per epoch — enough to see the curve without burying the epoch results.
LOGGING_STEPS = 20

EVAL_NUM_BEAMS = 1

# Weight of the auxiliary table-classification loss: L = L_seq2seq + λ·L_tables.
# Fixed in advance rather than tuned, so no choice here was made by looking at test.
# Set to 0 to train the plain baseline — the head is then neither trained nor saved.
AUX_LOSS_WEIGHT = float(os.environ.get("BANGLASQL_AUX_WEIGHT", "0.5"))

# BANGLASQL_SMOKE=1 trains on a handful of pairs for one epoch. It checks the wiring on a
# CPU — that the auxiliary loss is computed and the head is saved — and nothing else.
SMOKE = os.environ.get("BANGLASQL_SMOKE") == "1"


class BanglaSQLDataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, config: dict):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.config    = config

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        model_inputs = self.tokenizer(
            format_input(pair["bangla_question"], self.config),
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
        # Which tables the gold query reads, as a multi-hot vector in TABLES order.
        model_inputs["table_labels"] = encode_tables(pair["tables_used"])
        return model_inputs


class TableLabelCollator:
    """Pads the seq2seq fields as before and stacks the table labels beside them.

    `table_labels` is taken out before DataCollatorForSeq2Seq runs: it has a fixed length
    of len(TABLES), so there is nothing to pad, and the seq2seq collator should not be
    asked to guess what an unfamiliar key means.
    """

    def __init__(self, seq2seq_collator):
        self.seq2seq_collator = seq2seq_collator

    def __call__(self, features):
        table_labels = [f.pop("table_labels") for f in features]
        batch = self.seq2seq_collator(features)
        batch["table_labels"] = torch.tensor(table_labels, dtype=torch.float)
        return batch


class AuxSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer whose training loss adds λ·BCE(table head, tables used).

    The head reads the encoder states T5 already returns alongside the loss, so the
    auxiliary objective costs one extra matrix product per batch, not a second forward.
    """

    def __init__(self, *args, aux_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_weight = aux_weight
        self.bce = torch.nn.BCEWithLogitsLoss()
        self._aux_losses = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # A missing key here means the Trainer's column filter dropped it before the batch
        # arrived, and training would carry on without the auxiliary loss and without an
        # error. Failing loudly is the only way that shows up.
        if "table_labels" not in inputs:
            raise KeyError("table_labels missing from the batch — was "
                           "remove_unused_columns left at its default of True?")
        table_labels = inputs.pop("table_labels")

        outputs = model(**inputs)
        loss = outputs.loss
        if self.aux_weight > 0:
            pooled = pool(outputs.encoder_last_hidden_state, inputs["attention_mask"])
            aux = self.bce(model.table_head(pooled), table_labels.to(pooled.dtype))
            loss = loss + self.aux_weight * aux
            self._aux_losses.append(aux.detach().item())
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        # Evaluation generates SQL and reports the seq2seq loss only; T5's forward does
        # not accept table_labels, so they have to leave the batch first.
        inputs.pop("table_labels", None)
        return super().prediction_step(model, inputs, prediction_loss_only,
                                       ignore_keys=ignore_keys, **gen_kwargs)

    def log(self, logs, *args, **kwargs):
        # Shown beside the usual {'loss': ...} line, so the head can be seen learning.
        if self._aux_losses and "loss" in logs:
            logs["aux_loss"] = round(sum(self._aux_losses) / len(self._aux_losses), 5)
            self._aux_losses = []
        super().log(logs, *args, **kwargs)


class WarmupEarlyStoppingCallback(EarlyStoppingCallback):
    """EarlyStoppingCallback that ignores evaluations before `min_epochs`.

    Accuracy reads 0.0 for the first epochs while the model is still learning SQL
    syntax, so plain patience would burn through and stop long before convergence.
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


def compute_metrics(eval_preds, tokenizer, con):
    """Exact match and execution accuracy against the database."""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds  = np.asarray(preds)
    labels = np.asarray(labels)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    preds  = np.where((preds != -100) & (preds >= 0), preds, pad_id)
    labels = np.where((labels != -100) & (labels >= 0), labels, pad_id)

    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    total = len(decoded_preds)
    if total == 0:
        return {"exact_match": 0.0, "execution_accuracy": 0.0}

    exact = executed = 0
    for pred, gold in zip(decoded_preds, decoded_labels):
        exact += normalize_sql(pred) == normalize_sql(gold)
        gold_rows, _ = run_sql(con, gold)
        pred_rows, _ = run_sql(con, pred)
        executed += results_match(gold_rows, pred_rows, gold)

    return {
        "exact_match":        round(exact / total, 4),
        "execution_accuracy": round(executed / total, 4),
    }


def main():
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"Database not found: {DB_PATH}\nRun: python create_database.py")

    logger.info("=" * 60)
    logger.info("BanglaSQL — Phase 3: Training")
    logger.info("=" * 60)
    logger.info(f"Model      : {MODEL_NAME}")
    logger.info(f"Input      : question{' + schema' if TRAIN_CONFIG['include_schema'] else ' only'}")
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

    # Registered as a submodule, so the optimizer, device placement, checkpointing and
    # load_best_model_at_end all include it without further plumbing.
    if AUX_LOSS_WEIGHT > 0:
        model.table_head = TableHead(model.config.d_model, len(TABLES))
        logger.info(f"Table head : {sum(p.numel() for p in model.table_head.parameters()):,} "
                    f"parameters, λ = {AUX_LOSS_WEIGHT}")
    else:
        logger.info("Table head : off (AUX_LOSS_WEIGHT = 0) — plain baseline")

    logger.info("\nLoading datasets...")
    train_pairs = load_split("train")
    dev_pairs   = load_split("dev")
    if SMOKE:
        train_pairs, dev_pairs = train_pairs[:32], dev_pairs[:16]
        logger.info("  [SMOKE] 32 train / 16 dev pairs, 1 epoch — wiring check only")
    logger.info(f"  Train: {len(train_pairs)} pairs")
    logger.info(f"  Dev  : {len(dev_pairs)} pairs")

    train_dataset = BanglaSQLDataset(train_pairs, tokenizer, TRAIN_CONFIG)
    dev_dataset   = BanglaSQLDataset(dev_pairs, tokenizer, TRAIN_CONFIG)

    data_collator = TableLabelCollator(DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    ))

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
        "metric_for_best_model": "eval_execution_accuracy",
        "greater_is_better": True,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "predict_with_generate": True,
        "generation_max_length": MAX_TARGET_LENGTH,
        "generation_num_beams": EVAL_NUM_BEAMS,
        "logging_dir": LOGS_DIR,
        "logging_steps": LOGGING_STEPS,
        "report_to": "none",
        # Without this the Trainer installs its progress-bar callback, which repaints a
        # per-step bar with an ETA. Disabling it swaps in the plain printer, which emits
        # one dict per logging_steps and one per evaluation — the loss line and the
        # epoch result, and nothing else.
        "disable_tqdm": True,
        "seed": SEED,
        "data_seed": SEED,
        "fp16": False,
        "label_smoothing_factor": LABEL_SMOOTHING,
        # The default (True) wraps the collator in a filter that drops every batch key
        # T5.forward does not accept. table_labels is such a key, so training would run
        # without the auxiliary loss and without any error. AuxSeq2SeqTrainer consumes it.
        "remove_unused_columns": False,
    }
    if SMOKE:
        # 32 pairs at batch 8 is 4 steps, so log every step or no loss line ever prints.
        training_kwargs["num_train_epochs"] = 1
        training_kwargs["logging_steps"] = 1

    if "eval_strategy" in args_sig:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in args_sig:
        training_kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in args_sig:
        training_kwargs["save_strategy"] = "epoch"

    training_args = Seq2SeqTrainingArguments(
        **{k: v for k, v in training_kwargs.items() if k in args_sig}
    )

    con = open_readonly_db()
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": data_collator,
        "compute_metrics": lambda p: compute_metrics(p, tokenizer, con),
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

    trainer = AuxSeq2SeqTrainer(**trainer_kwargs, aux_weight=AUX_LOSS_WEIGHT)

    logger.info("\nStarting training...")
    trainer.train()

    best_model_dir = os.path.join(CHECKPOINTS, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    # The head is saved on its own and then detached, so model.safetensors holds a plain
    # T5 that AutoModelForSeq2SeqLM.from_pretrained loads with no unexpected keys.
    if hasattr(model, "table_head"):
        save_head(model.table_head, best_model_dir)
        del model.table_head
        logger.info(f"Table head saved to: {os.path.join(best_model_dir, 'table_head.pt')}")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    save_model_config(TRAIN_CONFIG, best_model_dir)
    logger.info(f"Best model saved to: {best_model_dir}")

    history_path = os.path.join(LOGS_DIR, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=float)
    logger.info(f"Training history saved to: {history_path}")

    logger.info("\nFinal dev set evaluation...")
    dev_metrics = trainer.evaluate()
    con.close()
    with open(os.path.join(LOGS_DIR, "dev_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2, default=float)

    best_ex = dev_metrics.get("eval_execution_accuracy", 0)
    logger.info("\n" + "=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Dev execution accuracy : {best_ex}")
    logger.info(f"  Dev exact match        : {dev_metrics.get('eval_exact_match', 0)}")
    logger.info(f"  Best model at          : {best_model_dir}")
    logger.info("=" * 60)
    logger.info("Next: python evaluate.py   (beam + execution-guided decoding on the test split)")

    if isinstance(best_ex, (int, float)) and best_ex < 0.20:
        logger.warning(
            "\n[FALLBACK TRIGGER] Dev execution accuracy < 20%. Consider setting "
            "\"model_name\" to \"google/mt5-small\" in data/train_config.json and retraining."
        )


if __name__ == "__main__":
    main()
