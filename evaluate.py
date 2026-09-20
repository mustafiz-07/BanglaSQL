"""
BanglaSQL — Phase 4: Evaluation Harness

Generates SQL for a dataset split with the fine-tuned checkpoint and reports, for
both plain top-1 beam decoding and execution-guided decoding:
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
    logs/<split>_results.json      metrics (headline = execution-guided)
    logs/<split>_predictions.json  per-example predictions for error analysis
    --tag NAME inserts NAME into both filenames, so the constrained and unconstrained
    decoders can be run back to back without overwriting each other.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from common import (
    DB_PATH, MODEL_CONFIG_NAME, format_input, load_config, load_split,
    normalize_sql, open_readonly_db, pick_executable, results_match, run_sql,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")


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

def generate_candidates(model, tokenizer, questions, config, num_beams, batch_size=16,
                        decoder=None):
    """Return, per question, all `num_beams` beam outputs in model-score order.

    With `decoder` set, generation is restricted to schema-valid continuations at every
    step (see constrained_decode). That is a different mechanism from the execution-guided
    reranking in pick_executable: reranking chooses among finished beams and is helpless
    when all of them are invalid, which is what the 52 malformed predictions in run 7 were.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    gen_kwargs = {
        "max_length": int(config["max_target_length"]),
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
    }
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True
    if decoder is not None:
        gen_kwargs["prefix_allowed_tokens_fn"] = decoder.prefix_fn()

    candidates = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        enc = tokenizer(
            [format_input(q, config) for q in batch],
            max_length=int(config["max_input_length"]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        candidates.extend(decoded[i:i + num_beams] for i in range(0, len(decoded), num_beams))
        print(f"  generated {min(start + batch_size, len(questions))}/{len(questions)}", end="\r")
    print()
    return candidates


# ── Scoring ────────────────────────────────────────────────────────────────────

def score(pairs, preds, con):
    """Compute every metric for one list of predictions. Returns (summary, records)."""
    failures = Counter()
    by_difficulty = defaultdict(lambda: {"n": 0, "exec": 0, "em": 0})
    by_query_type = defaultdict(lambda: {"n": 0, "exec": 0})
    component_stats = defaultdict(lambda: {"applicable": 0, "correct": 0})
    records = []
    n_exec = n_em = n_valid = 0

    for pair, pred_sql in zip(pairs, preds):
        gold_sql = pair["sql_query"]
        gold_rows, _ = run_sql(con, gold_sql)
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
            "template_id":      pair.get("template_id"),
            "difficulty":       diff,
            "query_type":       qtype,
            "bangla_question":  pair["bangla_question"],
            "gold_sql":         gold_sql,
            "pred_sql":         pred_sql,
            "valid":            valid,
            "exact_match":      exact,
            "execution_match":  correct,
            "error":            pred_err,
            "failure_category": category,
        })

    total = len(pairs)
    summary = {
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
    return summary, records


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate BanglaSQL on a dataset split")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "checkpoints", "best_model"))
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    parser.add_argument("--constrained", action="store_true",
                        help="restrict generation to schema-valid continuations at every step")
    parser.add_argument("--constrained-fallback", action="store_true",
                        help="generate unconstrained first, and re-generate with the "
                             "constraint only for questions where no beam executes")
    parser.add_argument("--tag", default="",
                        help="suffix for the output filenames, so two decoders can be "
                             "compared without overwriting each other")
    args = parser.parse_args()

    # The two constraint modes are alternatives. Accepting both silently ran fallback
    # while the results file recorded "constrained": true and a decoding label naming
    # both — a mislabelled arm, which is the same hazard that made run 8's first ablation
    # look like a result when nothing had been constrained at all.
    if args.constrained and args.constrained_fallback:
        raise SystemExit("--constrained and --constrained-fallback are alternatives: the "
                         "first constrains every question, the second only those with no "
                         "executable beam. Pick one.")

    if not os.path.isdir(args.model):
        raise SystemExit(f"Checkpoint not found: {args.model}\nTrain a model first (python train.py).")
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"Database not found: {DB_PATH}\nRun: python create_database.py")

    config = load_config(args.model)
    if not os.path.exists(os.path.join(args.model, MODEL_CONFIG_NAME)):
        print(f"[!] {args.model} has no {MODEL_CONFIG_NAME}; assuming data/train_config.json "
              f"matches how it was trained (include_schema={config['include_schema']}).")

    pairs = load_split(args.split)
    if args.limit:
        pairs = pairs[:args.limit]

    print("=" * 60)
    print(f"BanglaSQL — Phase 4: Evaluation ({args.split} split, {len(pairs)} examples)")
    print("=" * 60)
    print(f"Checkpoint: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    decoder = None
    if args.constrained or args.constrained_fallback:
        from constrained_decode import ConstrainedDecoder
        decoder = ConstrainedDecoder(tokenizer, max_length=int(config["max_target_length"]))
        if not decoder.schema.table_names:
            raise SystemExit("Schema is empty — is banglasql.db present? "
                             "Constrained decoding would silently do nothing.")
        print(f"\nSchema-constrained decoding: {len(decoder.schema.table_names)} tables, "
              f"{len(decoder.schema.all_columns)} columns")

    questions = [p["bangla_question"] for p in pairs]
    print("\nGenerating SQL...")
    candidates = generate_candidates(
        model, tokenizer, questions, config, args.num_beams,
        decoder=None if args.constrained_fallback else decoder,
    )

    stuck = []
    if args.constrained_fallback:
        # Constraining every question costs accuracy, because execution-guided reranking
        # was using invalidity as a free correctness signal: when the top beams are
        # malformed, reranking falls through to a lower beam that is often right. Making
        # those beams schema-valid promotes semantically wrong queries over correct ones —
        # 10 of run 8's regressions were valid-but-wrong queries displacing a correct
        # lower beam. So constrain only where reranking has nothing to fall through to.
        probe = open_readonly_db()
        stuck = [i for i, beams in enumerate(candidates)
                 if all(run_sql(probe, sql)[1] is not None for sql in beams)]
        probe.close()
        print(f"\n{len(stuck)} question(s) have no executable beam — re-generating those "
              f"with the constraint")
        if stuck:
            redone = generate_candidates(
                model, tokenizer, [questions[i] for i in stuck], config, args.num_beams,
                decoder=decoder,
            )
            for i, beams in zip(stuck, redone):
                candidates[i] = beams

    if decoder is not None:
        print("  " + decoder.report())
        # A constraint that never fired is a wiring failure, not a result. Run 8's first
        # ablation returned byte-identical predictions for both arms because main() built
        # no decoder at all, and only the label in the results file differed.
        #
        # In fallback mode the decoder is only invoked for questions with no executable
        # beam, so zero constrained steps is the *success* case when there were none —
        # checking it there would abort a run that worked. Assert only when the decoder
        # was actually asked to generate something.
        expected_to_fire = args.constrained or (args.constrained_fallback and stuck)
        if expected_to_fire and decoder.stats["identifier_constrained"] == 0:
            raise SystemExit("Constrained decoding never restricted a single step — "
                             "the constraint is not reaching generation.")

    print("Executing queries against the database...")
    con = open_readonly_db()
    top1_preds   = [c[0] for c in candidates]
    guided_preds = [pick_executable(c, con) for c in candidates]
    top1, _ = score(pairs, top1_preds, con)
    guided, records = score(pairs, guided_preds, con)
    con.close()

    for record, beams in zip(records, candidates):
        record["top1_sql"] = beams[0]
        record["used_lower_beam"] = record["pred_sql"] != beams[0]

    total = len(pairs)
    results = {
        "split": args.split,
        "checkpoint": args.model,
        "num_examples": total,
        "num_beams": args.num_beams,
        "decoding": ("execution_guided"
                     + ("+schema_constrained" if args.constrained else "")
                     + ("+schema_constrained_fallback" if args.constrained_fallback else "")),
        "constrained": bool(args.constrained),
        "constrained_fallback": bool(args.constrained_fallback),
        **guided,
        "top1_metrics": top1["metrics"],
    }

    os.makedirs(LOGS_DIR, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    results_path = os.path.join(LOGS_DIR, f"{args.split}{tag}_results.json")
    preds_path   = os.path.join(LOGS_DIR, f"{args.split}{tag}_predictions.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"RESULTS — {args.split} split ({total} examples)")
    print("=" * 60)
    print(f"  {'':20s} {'top-1 beam':>12s} {'exec-guided':>12s}")
    for key, label in [("execution_accuracy", "Execution accuracy"),
                       ("exact_match", "Exact match"),
                       ("validity_rate", "Validity rate")]:
        print(f"  {label:20s} {top1['metrics'][key]:>12.1%} {guided['metrics'][key]:>12.1%}")
    print(f"  (lower beam used for {sum(r['used_lower_beam'] for r in records)} questions)")

    print("\n  By difficulty (exec-guided):")
    for d, s in guided["by_difficulty"].items():
        print(f"    {d:8s} n={s['n']:4d}  exec={s['execution_accuracy']:.1%}  em={s['exact_match']:.1%}")

    print("\n  Component accuracy (exec-guided):")
    for name, s in guided["component_accuracy"].items():
        if s["accuracy"] is not None:
            print(f"    {name:10s} n={s['applicable']:4d}  {s['accuracy']:.1%}")

    if guided["failure_categories"]:
        print("\n  Failure categories (exec-guided):")
        for category, count in guided["failure_categories"].items():
            print(f"    {category:22s} {count:4d}  ({count/total:.1%})")

    print(f"\n  Saved: {results_path}")
    print(f"  Saved: {preds_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
