"""
BanglaSQL — auxiliary table-classification head.

A small classifier on top of the T5 encoder that predicts, from the Bangla question alone,
which of the six tables the SQL query uses. It is multi-label: "Ayesha Ahmed কোন কোন কোর্সে
নথিভুক্ত?" uses students, enrollments and courses at once.

    question ─► T5 encoder ─► hidden (B, L, d) ─► masked mean-pool ─► dropout ─► Linear(d → 6)

It serves twice:

    training   an auxiliary loss, L = L_seq2seq + λ·L_tables, so the encoder is pushed to
               represent which tables a question is about
    inference  a tie-breaker between beams — prefer the executable beam whose FROM/JOIN
               tables match the prediction (common.pick_executable)

Why tables: `wrong_table` is the largest failure category on the test set, and those queries
execute, so execution-guided decoding alone cannot reject them. Labels cost nothing — every
dataset pair already carries `tables_used`, and it matches the SQL in all 241 templates.

The head is saved beside the model as `table_head.pt`, never inside `model.safetensors`, so a
checkpoint still loads with plain `AutoModelForSeq2SeqLM.from_pretrained`. `load_head`
returns None when the file is absent, and callers then behave exactly as before.
"""

import os

import torch
from torch import nn

from common import format_input

#: Label order. Fixed rather than read from the database, and saved with the head, so a
#: head trained today still decodes correctly if the schema listing order ever changes.
TABLES = ["attendance", "courses", "departments", "enrollments", "instructors", "students"]

HEAD_FILE = "table_head.pt"

#: A table is predicted when its sigmoid probability reaches this.
THRESHOLD = 0.5


class TableHead(nn.Module):
    """Dropout + one linear layer over the pooled encoder state — 4.6K parameters."""

    def __init__(self, d_model: int, num_tables: int = len(TABLES), dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_tables)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Pooled encoder state (B, d) -> one logit per table (B, num_tables)."""
        return self.classifier(self.dropout(pooled))


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean of the encoder states over real tokens only.

    Padding is excluded: in a padded batch a short question would otherwise be averaged
    with the pad positions, and its representation would depend on which longer
    question it happened to share a batch with.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def encode_tables(tables_used) -> list[float]:
    """Multi-hot label in TABLES order."""
    used = {t.lower() for t in tables_used}
    return [1.0 if table in used else 0.0 for table in TABLES]


def save_head(head: TableHead, model_dir: str):
    torch.save({"state_dict": head.state_dict(),
                "tables": TABLES,
                "d_model": head.classifier.in_features},
               os.path.join(model_dir, HEAD_FILE))


def load_head(model_dir: str) -> TableHead | None:
    """The head saved with a checkpoint, or None if this checkpoint was trained without one."""
    path = os.path.join(model_dir, HEAD_FILE)
    if not os.path.exists(path):
        return None
    saved = torch.load(path, map_location="cpu")
    if saved["tables"] != TABLES:
        raise ValueError(f"{path} was trained with table order {saved['tables']}, "
                         f"but this code uses {TABLES}")
    head = TableHead(saved["d_model"])
    head.load_state_dict(saved["state_dict"])
    head.eval()
    return head


@torch.no_grad()
def predict_tables(model, head: TableHead, tokenizer, questions: list[str], config: dict,
                   batch_size: int = 32) -> list[frozenset]:
    """The table set the head predicts for each question.

    Runs the encoder only. Questions are ~20 tokens, so this second encoder pass costs a
    small fraction of beam search.
    """
    device = next(model.parameters()).device
    head.to(device).eval()
    predictions = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        enc = tokenizer([format_input(q, config) for q in batch],
                        max_length=int(config["max_input_length"]),
                        truncation=True, padding=True, return_tensors="pt").to(device)
        hidden = model.get_encoder()(input_ids=enc["input_ids"],
                                     attention_mask=enc["attention_mask"]).last_hidden_state
        probs = torch.sigmoid(head(pool(hidden, enc["attention_mask"])))
        for row in probs.tolist():
            predictions.append(frozenset(t for t, p in zip(TABLES, row) if p >= THRESHOLD))
    return predictions
