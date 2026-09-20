"""
BanglaSQL — schema-constrained decoding ablation.

Runs the same checkpoint through both decoders and writes the comparison table.

    python run_ablation.py                       # test split, checkpoints/best_model
    python run_ablation.py --limit 50            # quick check before the full run
    python run_ablation.py --split dev

Because the constraint is applied at inference only, the two runs share weights, data and
random seed — the sole difference is whether generation may leave the schema. That makes
this a controlled ablation rather than two separate experiments.

Outputs:
    logs/test_baseline_results.json      execution-guided reranking only
    logs/test_constrained_results.json   + schema-constrained generation
    logs/ablation.md                     the comparison table, ready to paste into a report
"""

import argparse
import json
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

ARMS = [
    ("baseline",    "execution-guided reranking", []),
    ("constrained", "+ schema-constrained generation", ["--constrained"]),
]


def run_arm(tag: str, extra: list[str], args) -> dict:
    cmd = [sys.executable, os.path.join(BASE_DIR, "evaluate.py"),
           "--split", args.split, "--model", args.model,
           "--num-beams", str(args.num_beams), "--tag", tag] + extra
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    print(f"\n{'=' * 70}\n{tag}: {' '.join(cmd[2:])}\n{'=' * 70}")
    start = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode != 0:
        raise SystemExit(f"{tag} failed with exit code {result.returncode}")

    path = os.path.join(LOGS_DIR, f"{args.split}_{tag}_results.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["wall_seconds"] = round(time.time() - start, 1)
    return data


def pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def delta(new, old) -> str:
    if new is None or old is None:
        return "—"
    diff = 100 * (new - old)
    return f"{diff:+.1f}" if abs(diff) >= 0.05 else "0.0"


def build_table(results: dict, split: str) -> str:
    base, con = results["baseline"], results["constrained"]
    lines = []

    lines.append(f"# Schema-constrained decoding — ablation on the {split} split\n")
    lines.append(f"Checkpoint: `{base['checkpoint']}`  ·  {base['num_examples']} examples  "
                 f"·  {base['num_beams']} beams\n")
    lines.append("Same weights, same data, same seed. The only difference is whether "
                 "generation is permitted to leave the schema.\n")

    lines.append("| metric | baseline | constrained | Δ |")
    lines.append("|---|---|---|---|")
    for label, key in [("validity rate", "validity_rate"),
                       ("execution accuracy", "execution_accuracy"),
                       ("exact match", "exact_match")]:
        b, c = base["metrics"].get(key), con["metrics"].get(key)
        lines.append(f"| **{label}** | {pct(b)} | {pct(c)} | {delta(c, b)} |")
    for label, key in [("execution accuracy (top-1 beam)", "execution_accuracy"),
                       ("validity rate (top-1 beam)", "validity_rate")]:
        b = base.get("top1_metrics", {}).get(key)
        c = con.get("top1_metrics", {}).get(key)
        lines.append(f"| {label} | {pct(b)} | {pct(c)} | {delta(c, b)} |")
    lines.append(f"| wall time | {base['wall_seconds']}s | {con['wall_seconds']}s | |")

    lines.append("\n## Failure categories\n")
    cats = sorted(set(base["failure_categories"]) | set(con["failure_categories"]))
    lines.append("| category | baseline | constrained | Δ |")
    lines.append("|---|---|---|---|")
    for cat in sorted(cats, key=lambda c: -base["failure_categories"].get(c, 0)):
        b = base["failure_categories"].get(cat, 0)
        c = con["failure_categories"].get(cat, 0)
        lines.append(f"| `{cat}` | {b} | {c} | {c - b:+d} |")

    lines.append("\n## Component accuracy\n")
    lines.append("| component | baseline | constrained | Δ |")
    lines.append("|---|---|---|---|")
    for name in sorted(base["component_accuracy"]):
        b = base["component_accuracy"][name]["accuracy"]
        c = con["component_accuracy"].get(name, {}).get("accuracy")
        lines.append(f"| `{name}` | {pct(b)} | {pct(c)} | {delta(c, b)} |")

    # The claim constrained decoding actually guarantees is validity; correctness is a
    # separate matter, so say which one moved.
    mal_b = base["failure_categories"].get("malformed_sql", 0)
    mal_c = con["failure_categories"].get("malformed_sql", 0)
    lines.append(f"\n## Reading\n")
    lines.append(f"- Malformed SQL: **{mal_b} → {mal_c}** "
                 f"({mal_b - mal_c} queries no longer fail to execute).")
    lines.append(f"- Validity is the effect constrained decoding guarantees. Execution "
                 f"accuracy moves only where the next-best schema-valid continuation "
                 f"happens to be the correct one, so it is expected to move less.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "checkpoints", "best_model"))
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(
            f"No checkpoint at {args.model}\n"
            "Unzip the model downloaded from Colab so that this path contains "
            "config.json, model.safetensors, the tokenizer files and banglasql_config.json."
        )

    results = {tag: run_arm(tag, extra, args) for tag, _, extra in ARMS}

    table = build_table(results, args.split)
    os.makedirs(LOGS_DIR, exist_ok=True)
    out = os.path.join(LOGS_DIR, "ablation.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(table)

    print("\n" + "=" * 70)
    print(table)
    print("=" * 70)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
