"""
BanglaSQL — Phase 4: Evaluation Harness

Generates SQL for a dataset split with the fine-tuned checkpoint and reports:
  - execution accuracy (primary) — does the SQL return the gold result set?
  - exact match (secondary)      — normalized string equality
  - validity rate                — does the SQL execute at all?
  - component accuracy           — SELECT / WHERE / GROUP BY / ORDER BY / JOIN
  - breakdowns by difficulty and query_type, plus a failure-category table

Run:
    python evaluate.py                       # test split, checkpoints/best_model
    python evaluate.py --split dev
    python evaluate.py --model checkpoints/best_model --limit 50

Outputs:
    logs/<split>_results.json      metrics
    logs/<split>_predictions.json  per-example predictions for error analysis
"""

import argparse
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from common import DB_PATH, format_input, load_config, load_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

QUERY_TIMEOUT_STEPS = 2_000_000   # sqlite VM steps before a runaway query is aborted


# ── SQL utilities ──────────────────────────────────────────────────────────────

def normalize_sql(sql: str) -> str:
    """Collapse whitespace, drop the trailing semicolon, lowercase."""
    sql = re.sub(r"\s+", " ", str(sql).strip())
    return sql.rstrip(";").strip().lower()


def open_readonly_db(path: str) -> sqlite3.Connection:
    """Open the database read-only so a generated DROP/DELETE cannot do damage."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    steps = {"n": 0}

    def guard():
        steps["n"] += 1
        return 1 if steps["n"] > QUERY_TIMEOUT_STEPS else 0

    con.set_progress_handler(guard, 1000)
    return con


def run_sql(con: sqlite3.Connection, sql: str):
    """Execute SQL, returning (rows, error). rows is None when execution failed."""
    if not normalize_sql(sql).startswith("select"):
        return None, "not a SELECT statement"
    try:
        return con.execute(sql).fetchall(), None
    except Exception as exc:
        return None, str(exc)


def results_match(gold_rows, pred_rows, gold_sql: str) -> bool:
    """Compare result sets.

    Row order only matters when the gold query has an ORDER BY; otherwise SQLite
    is free to return rows in any order and comparing ordered lists would fail
    semantically correct predictions.
    """
    if gold_rows is None or pred_rows is None:
        return False
    if len(gold_rows) != len(pred_rows):
        return False
    if "order by" in normalize_sql(gold_sql):
        return gold_rows == pred_rows
    return Counter(map(repr, gold_rows)) == Counter(map(repr, pred_rows))


# ── Component extraction ───────────────────────────────────────────────────────

CLAUSE_BOUNDARY = r"(?: from | where | group by | having | order by | limit |$)"

COMPONENT_PATTERNS = {
    "select":   r"^select (.*?)(?= from |$)",
    "where":    r" where (.*?)(?=" + CLAUSE_BOUNDARY + ")",
    "group_by": r" group by (.*?)(?=" + CLAUSE_BOUNDARY + ")",
    "having":   r" having (.*?)(?=" + CLAUSE_BOUNDARY + ")",
    "order_by": r" order by (.*?)(?=" + CLAUSE_BOUNDARY + ")",
}


def extract_components(sql: str) -> dict:
    """Pull out comparable clause strings from a normalized SQL query."""
    norm = normalize_sql(sql)
    comps = {}
    for name, pattern in COMPONENT_PATTERNS.items():
        match = re.search(pattern, norm)
        comps[name] = match.group(1).strip() if match else None

    joins = re.findall(r"join \w+(?: \w+)? on [\w.]+ ?= ?[\w.]+", norm)
    comps["join"] = " | ".join(sorted(joins)) if joins else None

    tables = set(re.findall(r"(?:from|join) (\w+)", norm))
    comps["tables"] = " ".join(sorted(tables)) if tables else None

    aggs = re.findall(r"\b(count|sum|avg|min|max)\s*\(", norm)
    comps["aggregate"] = " ".join(sorted(aggs)) if aggs else None
    return comps


def categorize_failure(gold_sql: str, pred_sql: str, exec_error: str | None) -> str:
    """Assign a single failure category for the error-analysis table."""
    if exec_error:
        return "malformed_sql"
    g, p = extract_components(gold_sql), extract_components(pred_sql)
    for component, label in [
        ("tables", "wrong_table"),
        ("join", "wrong_join"),
        ("aggregate", "wrong_aggregate"),
        ("where", "wrong_filter"),
        ("group_by", "wrong_group_by"),
        ("order_by", "wrong_order_by"),
        ("select", "wrong_select_columns"),
    ]:
        if g[component] != p[component]:
            return label
    return "other"


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_sql(model, tokenizer, questions, schema, max_in, max_out, num_beams, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    predictions = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        enc = tokenizer(
            [format_input(q, schema) for q in batch],
            max_length=max_in,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_length=max_out,
                num_beams=num_beams,
                early_stopping=True,
            )
        predictions.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        print(f"  generated {min(start + batch_size, len(questions))}/{len(questions)}", end="\r")
    print()
    return predictions


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate BanglaSQL on a dataset split")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "checkpoints", "best_model"))
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(
            f"Checkpoint not found: {args.model}\nTrain a model first (python train.py)."
        )
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"Database not found: {DB_PATH}\nRun: python create_database.py")

    config = load_config()
    schema = config["schema_string"]

    pairs = load_split(args.split)
    if args.limit:
        pairs = pairs[:args.limit]

    print("=" * 60)
    print(f"BanglaSQL — Phase 4: Evaluation ({args.split} split, {len(pairs)} examples)")
    print("=" * 60)
    print(f"Checkpoint: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    print("\nGenerating SQL...")
    preds = generate_sql(
        model, tokenizer,
        [p["bangla_question"] for p in pairs],
        schema,
        int(config["max_input_length"]),
        int(config["max_target_length"]),
        args.num_beams,
    )

    print("Executing queries against the database...")
    con = open_readonly_db(DB_PATH)

    records = []
    failures = Counter()
    by_difficulty = defaultdict(lambda: {"n": 0, "exec": 0, "em": 0})
    by_query_type = defaultdict(lambda: {"n": 0, "exec": 0})
    component_stats = defaultdict(lambda: {"applicable": 0, "correct": 0})

    n_exec = n_em = n_valid = 0

    for pair, pred_sql in zip(pairs, preds):
        gold_sql = pair["sql_query"]
        gold_rows, gold_err = run_sql(con, gold_sql)
        pred_rows, pred_err = run_sql(con, pred_sql)

        valid = pred_err is None
        exact = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        correct = results_match(gold_rows, pred_rows, gold_sql)

        n_valid += valid
        n_em    += exact
        n_exec  += correct

        diff = pair.get("difficulty", "unknown")
        by_difficulty[diff]["n"] += 1
        by_difficulty[diff]["exec"] += correct
        by_difficulty[diff]["em"] += exact

        qtype = pair.get("query_type", "unknown")
        by_query_type[qtype]["n"] += 1
        by_query_type[qtype]["exec"] += correct

        gold_comps, pred_comps = extract_components(gold_sql), extract_components(pred_sql)
        for name in gold_comps:
            if gold_comps[name] is None and pred_comps[name] is None:
                continue
            component_stats[name]["applicable"] += 1
            component_stats[name]["correct"] += gold_comps[name] == pred_comps[name]

        category = None if correct else categorize_failure(gold_sql, pred_sql, pred_err)
        if category:
            failures[category] += 1

        records.append({
            "template_id":     pair.get("template_id"),
            "difficulty":      diff,
            "query_type":      qtype,
            "bangla_question": pair["bangla_question"],
            "gold_sql":        gold_sql,
            "pred_sql":        pred_sql,
            "valid":           valid,
            "exact_match":     exact,
            "execution_match": correct,
            "error":           pred_err,
            "failure_category": category,
        })

    con.close()
    total = len(pairs)

    results = {
        "split": args.split,
        "checkpoint": args.model,
        "num_examples": total,
        "num_beams": args.num_beams,
        "metrics": {
            "execution_accuracy": round(n_exec / total, 4),
            "exact_match":        round(n_em / total, 4),
            "validity_rate":      round(n_valid / total, 4),
        },
        "by_difficulty": {
            d: {
                "n": s["n"],
                "execution_accuracy": round(s["exec"] / s["n"], 4),
                "exact_match":        round(s["em"] / s["n"], 4),
            }
            for d, s in sorted(by_difficulty.items())
        },
        "by_query_type": {
            q: {"n": s["n"], "execution_accuracy": round(s["exec"] / s["n"], 4)}
            for q, s in sorted(by_query_type.items(), key=lambda kv: -kv[1]["n"])
        },
        "component_accuracy": {
            name: {
                "applicable": s["applicable"],
                "accuracy": round(s["correct"] / s["applicable"], 4) if s["applicable"] else None,
            }
            for name, s in sorted(component_stats.items())
        },
        "failure_categories": dict(failures.most_common()),
    }

    os.makedirs(LOGS_DIR, exist_ok=True)
    results_path = os.path.join(LOGS_DIR, f"{args.split}_results.json")
    preds_path   = os.path.join(LOGS_DIR, f"{args.split}_predictions.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"RESULTS — {args.split} split ({total} examples)")
    print("=" * 60)
    print(f"  Execution accuracy : {results['metrics']['execution_accuracy']:.1%}")
    print(f"  Exact match        : {results['metrics']['exact_match']:.1%}")
    print(f"  Validity rate      : {results['metrics']['validity_rate']:.1%}")

    print("\n  By difficulty:")
    for d, s in results["by_difficulty"].items():
        print(f"    {d:8s} n={s['n']:4d}  exec={s['execution_accuracy']:.1%}  em={s['exact_match']:.1%}")

    print("\n  Component accuracy:")
    for name, s in results["component_accuracy"].items():
        if s["accuracy"] is not None:
            print(f"    {name:10s} n={s['applicable']:4d}  {s['accuracy']:.1%}")

    if failures:
        print("\n  Failure categories:")
        for category, count in results["failure_categories"].items():
            print(f"    {category:22s} {count:4d}  ({count/total:.1%})")

    print(f"\n  Saved: {results_path}")
    print(f"  Saved: {preds_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
