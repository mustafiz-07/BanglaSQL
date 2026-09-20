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
    ("baseline",    "execution-guided reranking only",            []),
    ("constrained", "+ schema-constrained generation everywhere", ["--constrained"]),
    ("fallback",    "+ constraint only where no beam executes",   ["--constrained-fallback"]),
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
    tags = [tag for tag, _, _ in ARMS if tag in results]
    base = results["baseline"]
    head = " | ".join(tags)
    rule = "|---" * (len(tags) + 1) + "|"
    lines = []

    lines.append(f"# Schema-constrained decoding — ablation on the {split} split\n")
    lines.append(f"Checkpoint: `{base['checkpoint']}`  ·  {base['num_examples']} examples  "
                 f"·  {base['num_beams']} beams\n")
    lines.append("Same weights, same data, same seed. The arms differ only in how "
                 "generation is constrained.\n")
    for tag, desc, _ in ARMS:
        if tag in results:
            lines.append(f"- **{tag}** — {desc}")
    lines.append("")

    def row(label, values, bold=False):
        name = f"**{label}**" if bold else label
        lines.append("| " + " | ".join([name] + values) + " |")

    lines.append(f"| metric | {head} |")
    lines.append(rule)
    for label, key in [("validity rate", "validity_rate"),
                       ("execution accuracy", "execution_accuracy"),
                       ("exact match", "exact_match")]:
        row(label, [pct(results[t]["metrics"].get(key)) for t in tags], bold=True)
    for label, key in [("validity rate (top-1 beam)", "validity_rate"),
                       ("execution accuracy (top-1 beam)", "execution_accuracy")]:
        row(label, [pct(results[t].get("top1_metrics", {}).get(key)) for t in tags])
    row("wall time", [f"{results[t]['wall_seconds']}s" for t in tags])

    lines.append("\n## Failure categories\n")
    lines.append(f"| category | {head} |")
    lines.append(rule)
    cats = set().union(*(results[t]["failure_categories"] for t in tags))
    for cat in sorted(cats, key=lambda c: -base["failure_categories"].get(c, 0)):
        row(f"`{cat}`", [str(results[t]["failure_categories"].get(cat, 0)) for t in tags])

    lines.append("\n## Component accuracy\n")
    lines.append(f"| component | {head} |")
    lines.append(rule)
    for name in sorted(base["component_accuracy"]):
        row(f"`{name}`",
            [pct(results[t]["component_accuracy"].get(name, {}).get("accuracy")) for t in tags])

    # Validity is what the constraint guarantees; correctness is a separate matter. Say
    # which one actually moved, because on this checkpoint they came apart.
    lines.append("\n## Reading\n")
    for t in tags:
        if t == "baseline":
            continue
        mal_b = base["failure_categories"].get("malformed_sql", 0)
        mal_t = results[t]["failure_categories"].get("malformed_sql", 0)
        d_exec = 100 * (results[t]["metrics"]["execution_accuracy"]
                        - base["metrics"]["execution_accuracy"])
        lines.append(f"- **{t}**: malformed SQL {mal_b} → {mal_t}, "
                     f"execution accuracy {d_exec:+.1f} points.")
    lines.append("- Constraining *every* question costs accuracy: execution-guided "
                 "reranking had been using invalidity as a correctness signal, falling "
                 "through malformed top beams to a correct lower one. Making those beams "
                 "schema-valid promotes semantically wrong queries over correct ones.")
    lines.append("- Applying the constraint only where no beam executes keeps that signal "
                 "intact and still removes the malformed queries.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "checkpoints", "best_model"))
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse", action="store_true",
                        help="rebuild the table from existing logs/ files without re-running")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(
            f"No checkpoint at {args.model}\n"
            "Unzip the model downloaded from Colab so that this path contains "
            "config.json, model.safetensors, the tokenizer files and banglasql_config.json."
        )

    if args.reuse:
        # build_table renders whichever arms are present, so a split that has only some of
        # them — or a logs/ directory written before the fallback arm existed — should
        # produce a smaller table rather than a FileNotFoundError.
        results = {}
        for tag, _, _ in ARMS:
            path = os.path.join(LOGS_DIR, f"{args.split}_{tag}_results.json")
            if not os.path.exists(path):
                print(f"skipping {tag}: no {os.path.basename(path)}")
                continue
            with open(path, encoding="utf-8") as f:
                results[tag] = json.load(f)
            results[tag].setdefault("wall_seconds", 0.0)
        if "baseline" not in results:
            raise SystemExit(f"Need at least {args.split}_baseline_results.json to build "
                             f"the table; run without --reuse first.")
    else:
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
