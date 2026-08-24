"""
BanglaSQL — Phase 3: Preprocessing & Tokenizer Analysis

Run this LOCALLY before training to:
  1. Normalize all Bangla text (Unicode NFC)
  2. Check tokenizer vocabulary coverage for Bangla tokens
  3. Profile sequence lengths (question + schema) to set max_length

Usage:
    pip install transformers sentencepiece
    python preprocess_check.py

Outputs a report and saves normalized datasets back to data/.

== Changelog (Phase 3 fix pass) ==
- Tokenizer coverage analysis, model selection (BanglaT5 vs mT5 fallback), and
  max_input_length/max_target_length recommendations were all computed over
  train + dev + test combined. That lets statistics from the held-out test
  set leak into decisions made before training (which tokenizer/model to use,
  how aggressively to truncate) — a mild but real form of test-set peeking.
  Fixed by computing these over train + dev only. Unicode normalization is
  still applied and saved for all three splits, since that's just consistent
  preprocessing, not a decision informed by test-set content.
"""

import json
import os
import unicodedata
from collections import Counter

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_NAME = "csebuetnlp/banglat5"          # primary
FALLBACK   = "google/mt5-small"             # fallback

# Schema string used during training — same function will be used in train.py
SCHEMA_STRING = (
    "table: departments(dept_id, dept_name, building, phone) | "
    "table: instructors(instructor_id, first_name, last_name, email, dept_id, designation, joining_year) | "
    "table: students(student_id, first_name, last_name, email, dept_id, year_of_admission, cgpa) | "
    "table: courses(course_id, course_code, course_name, credits, dept_id, instructor_id, semester) | "
    "table: enrollments(enrollment_id, student_id, course_id, grade, grade_point) | "
    "table: attendance(attendance_id, student_id, course_id, date, status)"
)


# ── 1. Unicode Normalization ────────────────────────────────────────────────────

def normalize_bangla(text: str) -> str:
    """Apply NFC Unicode normalization to ensure consistent Bangla representation."""
    return unicodedata.normalize("NFC", str(text).strip())


def normalize_dataset_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    for pair in pairs:
        original = pair["bangla_question"]
        normalized = normalize_bangla(original)
        if original != normalized:
            pair["bangla_question"] = normalized
            pair["was_normalized"] = True
    return pairs


def save_normalized(pairs: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)


# ── 2. Tokenizer Analysis ──────────────────────────────────────────────────────

def analyze_tokenizer(tokenizer, pairs: list[dict], model_name: str):
    """
    Check:
    - What fraction of Bangla tokens are split into sub-word fragments
    - Whether Bangla script is in the vocabulary at all
    """
    print(f"\n  Model: {model_name}")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # Count Bangla characters in vocab
    bangla_vocab_tokens = [
        tok for tok in tokenizer.get_vocab()
        if any('\u0980' <= ch <= '\u09FF' for ch in tok)
    ]
    print(f"  Bangla-script tokens in vocab: {len(bangla_vocab_tokens):,} "
          f"({len(bangla_vocab_tokens)/tokenizer.vocab_size*100:.1f}%)")

    # Sample tokenization analysis
    sample_questions = [p["bangla_question"] for p in pairs[:50]]
    fragment_counts = []
    total_tokens    = []

    for q in sample_questions:
        tokens = tokenizer.tokenize(q)
        # Count sub-word fragments (those starting with ## or ▁ in SentencePiece)
        fragments = sum(1 for t in tokens if not t.startswith("▁") and t not in ["<s>", "</s>"])
        fragment_counts.append(fragments)
        total_tokens.append(len(tokens))

    avg_tokens    = sum(total_tokens) / len(total_tokens)
    avg_fragments = sum(fragment_counts) / len(fragment_counts)
    coverage = 1.0 - (avg_fragments / avg_tokens) if avg_tokens > 0 else 0

    print(f"  Avg tokens per question (sample of 50): {avg_tokens:.1f}")
    print(f"  Avg fragment sub-words per question:    {avg_fragments:.1f}")
    print(f"  Estimated Bangla coverage score:        {coverage:.1%}")

    return coverage


# ── 3. Sequence Length Profiling ───────────────────────────────────────────────

def profile_sequence_lengths(tokenizer, pairs: list[dict]):
    """
    Profile the distribution of input (question + schema) and output (SQL) lengths.
    Helps determine the right max_input_length and max_target_length.
    """
    input_lengths  = []
    output_lengths = []

    for pair in pairs:
        inp = f"translate Bangla to SQL: {normalize_bangla(pair['bangla_question'])} </s> {SCHEMA_STRING}"
        out = pair["sql_query"]

        inp_ids = tokenizer(inp, return_tensors=None)["input_ids"]
        out_ids = tokenizer(out, return_tensors=None)["input_ids"]

        input_lengths.append(len(inp_ids))
        output_lengths.append(len(out_ids))

    def stats(lengths, label):
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        p50 = lengths_sorted[n // 2]
        p90 = lengths_sorted[int(n * 0.90)]
        p95 = lengths_sorted[int(n * 0.95)]
        p99 = lengths_sorted[int(n * 0.99)]
        print(f"\n  {label}:")
        print(f"    min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)//n}")
        print(f"    p50={p50}, p90={p90}, p95={p95}, p99={p99}")
        return p95

    print("\n  Sequence Length Profile (over train + dev pairs):")
    in_p95  = stats(input_lengths,  "Input (question + schema)")
    out_p95 = stats(output_lengths, "Output (SQL query)")

    recommended_in  = min(512, max(128, ((in_p95  // 64) + 1) * 64))
    recommended_out = min(256, max(64,  ((out_p95 // 32) + 1) * 32))

    print(f"\n  Recommended max_input_length  : {recommended_in}")
    print(f"  Recommended max_target_length : {recommended_out}")
    print("  (rounded up to nearest power-of-2 boundary, capped at 512/256)")

    return recommended_in, recommended_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BanglaSQL — Phase 3: Preprocessing & Tokenizer Analysis")
    print("=" * 60)

    # 1. Normalize (applied and saved for ALL splits — this is just consistent
    #    preprocessing, not a decision informed by data content, so including
    #    test here is fine).
    print("\n[1/3] Unicode Normalization (NFC)...")
    split_pairs = {}
    normalized_counts = {s: 0 for s in ["train", "dev", "test"]}
    for split in ["train", "dev", "test"]:
        path = os.path.join(DATA_DIR, f"dataset_{split}.json")
        if not os.path.exists(path):
            print(f"  Skipping {split} — file not found. Run build_dataset.py first.")
            continue
        pairs = normalize_dataset_file(path)
        normalized_counts[split] = sum(1 for p in pairs if p.get("was_normalized"))
        save_normalized(pairs, path)
        split_pairs[split] = pairs
        print(f"  {split}: {len(pairs)} pairs, {normalized_counts[split]} normalized")

    if not split_pairs:
        print("No data found. Run build_dataset.py first.")
        return

    # Everything below this point informs training decisions (tokenizer/model
    # choice, max_length settings), so it must NOT see the test split — using
    # test-set statistics to pick hyperparameters is a form of leakage.
    analysis_pairs = split_pairs.get("train", []) + split_pairs.get("dev", [])
    if not analysis_pairs:
        print("No train/dev data found (only test present). Run build_dataset.py first.")
        return
    if "test" in split_pairs:
        print(f"\n  (Analysis below uses train+dev only — {len(analysis_pairs)} pairs. "
              f"Test split ({len(split_pairs['test'])} pairs) is excluded from here on "
              f"so hyperparameter choices can't leak information from it.)")

    # 2. Tokenizer analysis
    print("\n[2/3] Tokenizer Analysis...")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  transformers not installed. Run: pip install transformers sentencepiece")
        return

    results = {}
    for model_name in [MODEL_NAME, FALLBACK]:
        print(f"\n  Loading tokenizer: {model_name}")
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            coverage = analyze_tokenizer(tok, analysis_pairs, model_name)
            results[model_name] = {"tokenizer": tok, "coverage": coverage}
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            results[model_name] = {"tokenizer": None, "coverage": 0.0}

    # Verdict
    bt5_cov = results.get(MODEL_NAME, {}).get("coverage", 0)
    mt5_cov = results.get(FALLBACK, {}).get("coverage", 0)

    print("\n  -- Tokenizer Verdict --")
    if bt5_cov >= mt5_cov:
        chosen = MODEL_NAME
        print(f"  Use BanglaT5 ({MODEL_NAME}): better Bangla coverage ({bt5_cov:.1%} vs {mt5_cov:.1%})")
    else:
        chosen = FALLBACK
        print(f"  Use mT5-small ({FALLBACK}): better coverage ({mt5_cov:.1%} vs {bt5_cov:.1%})")

    chosen_tok = results[chosen]["tokenizer"]
    if chosen_tok is None:
        print(f"\n  [!] Could not load a tokenizer for either model (see errors above — "
              f"likely a network issue). Skipping length profiling and config save; "
              f"train.py will fall back to its built-in defaults.")
        print("\n" + "=" * 60)
        return

    # 3. Sequence length profiling with chosen tokenizer (train+dev only)
    print("\n[3/3] Sequence Length Profiling...")
    rec_in, rec_out = profile_sequence_lengths(chosen_tok, analysis_pairs)

    # Save recommendations to config file for use by train.py
    config = {
        "model_name":         chosen,
        "max_input_length":   rec_in,
        "max_target_length":  rec_out,
        "schema_string":      SCHEMA_STRING,
        "bt5_coverage":       round(bt5_cov, 4),
        "mt5_coverage":       round(mt5_cov, 4),
    }
    config_path = os.path.join(DATA_DIR, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n  Config saved to: {config_path}")

    print("\n" + "=" * 60)
    print("Preprocessing complete. Now upload to Colab/Kaggle and run train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
