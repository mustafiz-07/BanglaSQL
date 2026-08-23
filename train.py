"""
BanglaSQL — Phase 3: Training Script
Seq2Seq fine-tuning for Bangla → SQL generation.

Designed to run on Colab / Kaggle (free GPU).
Models: csebuetnlp/banglat5 (primary) | google/mt5-small (fallback)

== Quick Start on Colab ==
1. Upload the entire project folder (or clone from GitHub)
2. Run: !pip install -r requirements.txt
3. Run: !python preprocess_check.py
4. Run: !python train.py

== Fallback trigger ==
If BanglaT5 does not reach >20% execution accuracy on the dev set
after 10 epochs, change MODEL_NAME to "google/mt5-small" and retrain.
"""

import json
import os
import random
import unicodedata
import logging

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

# ── Load train config (from preprocess_check.py output) ──────────────────────
CONFIG_PATH = os.path.join(DATA_DIR, "train_config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, encoding="utf-8") as f:
        TRAIN_CONFIG = json.load(f)
    logger.info(f"Loaded train config from {CONFIG_PATH}")
else:
    # Sensible defaults if preprocess_check.py was not run
    logger.warning("train_config.json not found — using defaults. Run preprocess_check.py first.")
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
MAX_INPUT_LENGTH  = TRAIN_CONFIG["max_input_length"]
MAX_TARGET_LENGTH = TRAIN_CONFIG["max_target_length"]
SCHEMA_STRING     = TRAIN_CONFIG["schema_string"]

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Tune these based on dev set performance
BATCH_SIZE       = 8       # reduce to 4 if OOM on Colab
GRAD_ACCUM_STEPS = 2       # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
LEARNING_RATE    = 5e-4
NUM_EPOCHS       = 20
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
SAVE_TOTAL_LIMIT = 3       # keep only the 3 best checkpoints


# ── Dataset ───────────────────────────────────────────────────────────────────

def normalize_bangla(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def format_input(question: str, schema: str) -> str:
    """
    Schema linearization:
    Input format: "translate Bangla to SQL: <question> </s> <schema>"
    This prefix-based format works well with T5-family models.
    """
    return f"translate Bangla to SQL: {normalize_bangla(question)} </s> {schema}"


def load_split(split: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"dataset_{split}.json")
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
    - exact_match: % of predictions that exactly match gold SQL (token-level)
    """
    preds, labels = eval_preds

    # Replace -100 (padding token in labels) with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)

    # Strip whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    exact_match = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    exact_match_pct = exact_match / len(decoded_preds) if decoded_preds else 0.0

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
        logger.warning("No GPU detected — training will be extremely slow. Use Colab/Kaggle.")

    # Load tokenizer & model
    logger.info(f"\nLoading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Load datasets
    logger.info("\nLoading datasets...")
    train_pairs = load_split("train")
    dev_pairs   = load_split("dev")
    logger.info(f"  Train: {len(train_pairs)} pairs")
    logger.info(f"  Dev  : {len(dev_pairs)} pairs")

    train_dataset = BanglaSQLDataset(train_pairs, tokenizer, SCHEMA_STRING)
    dev_dataset   = BanglaSQLDataset(dev_pairs,   tokenizer, SCHEMA_STRING)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINTS,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        save_total_limit=SAVE_TOTAL_LIMIT,

        # Generation settings for evaluation
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,

        # Logging
        logging_dir=LOGS_DIR,
        logging_steps=10,
        report_to="none",   # set to "wandb" if you want W&B logging

        # Reproducibility
        seed=SEED,
        data_seed=SEED,

        # FP16 (only on GPU)
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4)
            # Stops if exact_match doesn't improve for 4 consecutive epochs
        ],
    )

    # Train
    logger.info("\nStarting training...")
    train_result = trainer.train()

    # Save final model
    best_model_dir = os.path.join(CHECKPOINTS, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info(f"Best model saved to: {best_model_dir}")

    # Save training metrics
    metrics = train_result.metrics
    metrics_path = os.path.join(LOGS_DIR, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to: {metrics_path}")

    # Final dev evaluation
    logger.info("\nFinal dev set evaluation...")
    dev_metrics = trainer.evaluate()
    dev_metrics_path = os.path.join(LOGS_DIR, "dev_metrics.json")
    with open(dev_metrics_path, "w") as f:
        json.dump(dev_metrics, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Best dev exact_match : {dev_metrics.get('eval_exact_match', 'N/A')}")
    logger.info(f"  Best model at        : {best_model_dir}")
    logger.info("=" * 60)

    # Fallback advice
    best_em = dev_metrics.get("eval_exact_match", 0)
    if best_em < 0.20:
        logger.warning(
            "\n[FALLBACK TRIGGER] Dev exact_match < 20%.\n"
            "Consider switching MODEL_NAME to 'google/mt5-small' and retraining.\n"
            "Edit train_config.json: set 'model_name' to 'google/mt5-small'"
        )


if __name__ == "__main__":
    main()
