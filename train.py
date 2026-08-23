"""
BanglaSQL — Phase 3: Training Script
Seq2Seq fine-tuning for Bangla → SQL generation.

Designed to run on Colab / Kaggle (free GPU) or local machine.
Models: csebuetnlp/banglat5 (primary) | google/mt5-small (fallback)

== Quick Start on Colab ==
1. Upload the entire project folder (or clone from GitHub)
2. Run: !pip install -r requirements_colab.txt -q
3. Run: !python build_dataset.py
4. Run: !python preprocess_check.py
5. Run: !python train.py

== Fallback trigger ==
If BanglaT5 does not reach >20% execution accuracy on the dev set
after 10 epochs, change MODEL_NAME to "google/mt5-small" in train_config.json and retrain.
"""

import json
import os
import random
import unicodedata
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

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
CHECKPOINTS   = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")

os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ── Helpers for Safe JSON Serialization ───────────────────────────────────────
def save_json_safe(data: dict, filepath: str):
    """Serialize metrics and config dictionaries safely, handling numpy types."""
    def convert(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=convert, ensure_ascii=False)


# ── Load train config (from preprocess_check.py output) ──────────────────────
CONFIG_PATH = os.path.join(DATA_DIR, "train_config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, encoding="utf-8") as f:
        TRAIN_CONFIG = json.load(f)
    logger.info(f"Loaded train config from {CONFIG_PATH}")
else:
    logger.warning("train_config.json not found — using defaults. Run preprocess_check.py for optimal tuning.")
    TRAIN_CONFIG = {
        "model_name":        "csebuetnlp/banglat5",
        "max_input_length":  256,
        "max_target_length": 128,
        "schema_string": (
            "table: departments(dept_id, dept_name, building, phone) | "
            "table: instructors(instructor_id, first_name, last_name, email, dept_id, designation, joining_year) | "
            "table: students(student_id, first_name, last_name, email, dept_id, year_of_admission, cgpa) | "
            "table: courses(course_id, course_code, course_name, credits, dept_id, instructor_id, semester) | "
            "table: enrollments(enrollment_id, student_id, course_id, grade, grade_point) | "
            "table: attendance(attendance_id, student_id, course_id, date, status)"
        ),
    }

MODEL_NAME        = TRAIN_CONFIG["model_name"]
MAX_INPUT_LENGTH  = int(TRAIN_CONFIG["max_input_length"])
MAX_TARGET_LENGTH = int(TRAIN_CONFIG["max_target_length"])
SCHEMA_STRING     = TRAIN_CONFIG["schema_string"]

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE       = 8       # Batch size per device
GRAD_ACCUM_STEPS = 2       # Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
LEARNING_RATE    = 5e-4
NUM_EPOCHS       = 20
WEIGHT_DECAY     = 0.01
SAVE_TOTAL_LIMIT = 3       # Keep only the 3 best checkpoints


# ── Dataset ───────────────────────────────────────────────────────────────────

def normalize_bangla(text: str) -> str:
    """Normalize text into NFC Unicode format."""
    return unicodedata.normalize("NFC", str(text).strip())


def format_input(question: str, schema: str) -> str:
    """
    Schema linearization prefix formatting.
    Input format: "translate Bangla to SQL: <question> </s> <schema>"
    """
    return f"translate Bangla to SQL: {normalize_bangla(question)} </s> {schema}"


def load_split(split: str) -> list[dict]:
    """Load train/dev/test split JSON. Auto-generates dataset if missing."""
    path = os.path.join(DATA_DIR, f"dataset_{split}.json")
    if not os.path.exists(path):
        logger.warning(f"{path} not found. Attempting to build dataset using build_dataset.py...")
        try:
            import build_dataset
            build_dataset.main()
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load {path}. Please run 'python build_dataset.py' first. Error: {e}"
            )

    with open(path, encoding="utf-8") as f:
        return json.load(f)


class BanglaSQLDataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, schema: str):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.schema    = schema

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        inp  = format_input(pair["bangla_question"], self.schema)
        tgt  = pair["sql_query"]

        model_inputs = self.tokenizer(
            inp,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )

        # Cross-version compatibility for target tokenization
        try:
            labels = self.tokenizer(
                text_target=tgt,
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding=False,
            )
        except TypeError:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    tgt,
                    max_length=MAX_TARGET_LENGTH,
                    truncation=True,
                    padding=False,
                )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_preds, tokenizer):
    """
    Compute:
    - exact_match: % of predictions that match gold SQL after whitespace normalization
    """
    preds, labels = eval_preds

    # If predictions are returned as a tuple (e.g. generation logits or tuple of ids)
    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert preds and labels to numpy array if not already
    preds  = np.array(preds)
    labels = np.array(labels)

    # Pad token id fallback
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Replace -100 / negative values in both preds and labels before decoding
    preds  = np.where((preds != -100) & (preds >= 0), preds, pad_id)
    labels = np.where((labels != -100) & (labels >= 0), labels, pad_id)

    decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)

    # Strip and clean whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    exact_match = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    total = len(decoded_preds)
    exact_match_pct = (exact_match / total) if total > 0 else 0.0

    return {"exact_match": round(exact_match_pct, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("BanglaSQL — Phase 3: Training")
    logger.info("=" * 60)
    logger.info(f"Model       : {MODEL_NAME}")
    logger.info(f"Max input   : {MAX_INPUT_LENGTH}")
    logger.info(f"Max target  : {MAX_TARGET_LENGTH}")
    logger.info(f"Batch size  : {BATCH_SIZE} (x{GRAD_ACCUM_STEPS} grad accum = {BATCH_SIZE*GRAD_ACCUM_STEPS} effective)")
    logger.info(f"Epochs      : {NUM_EPOCHS}")
    logger.info(f"LR          : {LEARNING_RATE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device      : {device}")
    if device == "cpu":
        logger.warning("No GPU detected — training will be slow. Use Colab/Kaggle T4 GPU.")
    else:
        torch.cuda.empty_cache()
        logger.info(f"GPU Name    : {torch.cuda.get_device_name(0)}")

    # Load tokenizer & model
    logger.info(f"\nLoading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Clean tie_word_embeddings warning if applicable
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load datasets
    logger.info("\nLoading datasets...")
    train_pairs = load_split("train")
    dev_pairs   = load_split("dev")
    logger.info(f"  Train: {len(train_pairs)} pairs")
    logger.info(f"  Dev  : {len(dev_pairs)} pairs")

    train_dataset = BanglaSQLDataset(train_pairs, tokenizer, SCHEMA_STRING)
    dev_dataset   = BanglaSQLDataset(dev_pairs,   tokenizer, SCHEMA_STRING)

    # Data collator (handles dynamic padding and label_pad_token_id=-100)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    # Inspect signatures for cross-version compatibility
    args_sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters

    training_kwargs = {
        "output_dir": CHECKPOINTS,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "load_best_model_at_end": True,
        "metric_for_best_model": "exact_match",
        "greater_is_better": True,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "predict_with_generate": True,
        "generation_max_length": MAX_TARGET_LENGTH,
        "logging_dir": LOGS_DIR,
        "logging_steps": 10,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "fp16": torch.cuda.is_available(),
    }

    # Handle evaluation strategy across all transformers versions
    if "eval_strategy" in args_sig:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in args_sig:
        training_kwargs["evaluation_strategy"] = "epoch"

    if "save_strategy" in args_sig:
        training_kwargs["save_strategy"] = "epoch"

    # Handle warmup parameter across all transformers versions
    if "warmup_steps" in args_sig:
        training_kwargs["warmup_steps"] = 50
    elif "warmup_ratio" in args_sig:
        training_kwargs["warmup_ratio"] = 0.1

    # Filter to only pass parameters accepted by this specific version
    valid_args = {k: v for k, v in training_kwargs.items() if k in args_sig}
    training_args = Seq2SeqTrainingArguments(**valid_args)

    # Initialize Trainer with dynamic parameter support
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": data_collator,
        "compute_metrics": lambda p: compute_metrics(p, tokenizer),
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=4)
        ],
    }

    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # Train
    logger.info("\nStarting training...")
    train_result = trainer.train()

    # Save best model
    best_model_dir = os.path.join(CHECKPOINTS, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info(f"Best model saved to: {best_model_dir}")

    # Save training metrics (numpy-safe)
    metrics = train_result.metrics
    metrics_path = os.path.join(LOGS_DIR, "train_metrics.json")
    save_json_safe(metrics, metrics_path)
    logger.info(f"Training metrics saved to: {metrics_path}")

    # Final dev evaluation
    logger.info("\nFinal dev set evaluation...")
    dev_metrics = trainer.evaluate()
    dev_metrics_path = os.path.join(LOGS_DIR, "dev_metrics.json")
    save_json_safe(dev_metrics, dev_metrics_path)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Best dev exact_match : {dev_metrics.get('eval_exact_match', dev_metrics.get('exact_match', 'N/A'))}")
    logger.info(f"  Best model at        : {best_model_dir}")
    logger.info("=" * 60)

    # Fallback advice
    best_em = dev_metrics.get("eval_exact_match", dev_metrics.get("exact_match", 0))
    if isinstance(best_em, (int, float)) and best_em < 0.20:
        logger.warning(
            "\n[FALLBACK TRIGGER] Dev exact_match < 20%.\n"
            "Consider switching MODEL_NAME to 'google/mt5-small' in data/train_config.json and retraining."
        )


if __name__ == "__main__":
    main()
