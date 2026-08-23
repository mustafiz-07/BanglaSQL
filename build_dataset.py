"""
BanglaSQL — Dataset Builder (Phase 2)

Loads base templates, augments via rule-based paraphrasing, deduplicates,
and produces train/dev/test splits.

Augmentation approach:
  - No external API needed — purely rule-based synonym substitution and
    question reordering so results are fully reproducible.
  - Each template generates up to MAX_PARAPHRASES_PER_TEMPLATE variants.
  - Dedup by exact Bangla string match.
  - Splits preserve difficulty ratio and hold out 15% of query *patterns*
    entirely from train to test generalization.

Run:
    python build_dataset.py

Outputs:
    data/dataset_train.json
    data/dataset_dev.json
    data/dataset_test.json
    data/dataset_stats.json
"""

import json
import random
import re
import os
from copy import deepcopy

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEMPLATES_PATH = os.path.join(DATA_DIR, "templates.json")
MAX_PARAPHRASES_PER_TEMPLATE = 6  # cap to keep quality manageable
TRAIN_RATIO = 0.70
DEV_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Synonym dictionaries (Cholito Bangla) ─────────────────────────────────────

SYNONYMS = {
    # verbs / request phrases
    "দাও।":         ["দেখাও।", "বলো।", "প্রদর্শন করো।", "উপস্থাপন করো।"],
    "দেখাও।":       ["দাও।", "বলো।", "প্রদর্শন করো।"],
    "বলো।":         ["দাও।", "দেখাও।", "জানাও।"],
    "হিসাব করো।":  ["বের করো।", "নির্ণয় করো।", "গণনা করো।"],
    "বের করো।":    ["হিসাব করো।", "নির্ণয় করো।"],

    # query subjects
    "তালিকা দাও":       ["তথ্য দাও", "নাম দাও", "তথ্য দেখাও"],
    "তথ্য দাও":         ["তালিকা দাও", "বিস্তারিত দাও"],
    "নাম দাও":          ["তালিকা দাও", "তথ্য দাও"],
    "নাম ও CGPA দাও":   ["CGPA সহ নাম দাও", "নাম ও তাদের CGPA দেখাও"],

    # quantifiers / qualifiers
    "সকল":          ["সব", "সমস্ত", "যাবতীয়"],
    "সব":           ["সকল", "সমস্ত"],
    "সমস্ত":        ["সকল", "সব"],
    "যেসব":         ["যারা", "সেইসব", "ঐসব"],
    "যারা":         ["যেসব", "সেইসব"],
    "কারা":         ["কে কে", "কোন কোন"],

    # comparison phrases
    "বেশি":         ["বেশি হয়", "অধিক", "ঊর্ধ্বে"],
    "কম":           ["কমতি", "নিম্নে", "কম হয়"],
    "বেশি বা সমান": ["সমান বা তার বেশি", "কমপক্ষে"],
    "কমের কম":     ["সর্বনিম্ন", "অন্তত"],

    # sort / order phrases
    "অবরোহী ক্রমে সাজাও": ["বড় থেকে ছোট ক্রমে দেখাও", "নামতা ক্রমে সাজাও"],
    "আরোহী ক্রমে সাজাও":  ["ছোট থেকে বড় ক্রমে দেখাও", "উর্ধ্বক্রমে সাজাও"],
    "অবরোহী ক্রমে":       ["বড় থেকে ছোট ক্রমে", "নামতা অনুযায়ী"],

    # admission / year
    "ভর্তি হয়েছে":    ["ভর্তি নিয়েছে", "ভর্তি হয়েছিল"],
    "যোগ দিয়েছেন":  ["যোগদান করেছেন", "নিয়োগ পেয়েছেন"],
    "যোগ দেওয়া":    ["যোগদান করা"],

    # academic terms
    "নথিভুক্ত":    ["ভর্তি", "রেজিস্ট্রেশন করা"],
    "নথিভুক্ত আছে": ["ভর্তি আছে", "রেজিস্ট্রেশন করা আছে"],
    "পদমর্যাদার":  ["পদের", "পদবির"],
    "শিক্ষার্থী":  ["ছাত্রছাত্রী", "শিক্ষানবিস"],
    "শিক্ষক":     ["অধ্যাপক", "শিক্ষামণ্ডলী"],
    "বিভাগ":      ["ডিপার্টমেন্ট", "বিভাগীয়"],
    "কোর্স":      ["বিষয়", "পাঠ্যক্রম"],
    "সেমিস্টার":  ["সেশন", "পর্ব"],
    "উপস্থিত":    ["হাজির", "উপস্থিত ছিল"],
    "অনুপস্থিত":  ["অনুপস্থিত ছিল", "হাজির ছিল না", "গরহাজির"],

    # aggregate terms
    "গড়":        ["গড় মান", "সমগড়"],
    "মোট":       ["সর্বমোট", "মোট সংখ্যা"],
    "সর্বোচ্চ":  ["সবচেয়ে বেশি", "সর্বাধিক"],
    "সর্বনিম্ন": ["সবচেয়ে কম", "ন্যূনতম"],

    # question starters
    "কোন":       ["কোন কোন", "কোনটি"],
    "কতজন":     ["কত সংখ্যক", "মোট কতজন"],
    "কতটি":     ["কতগুলো", "কত সংখ্যক"],
}

# ── Prefix/suffix paraphrase templates ────────────────────────────────────────
# These are sentence-level rewrites applied on top of synonym substitution.
# Format: (prefix_to_add, suffix_to_add)  — both optional ("" = skip)

REWRITE_FRAMES = [
    # No change — base form
    ("", ""),
    # Add polite request prefix
    ("অনুগ্রহ করে, ", ""),
    # Add "আমাকে" (tell me)
    ("আমাকে জানাও — ", ""),
    # Rephrase as "কী?" style
    ("", " কী?"),
    # Add emphasis
    ("দয়া করে ", ""),
    # Indirect phrasing
    ("আমি জানতে চাই: ", ""),
]


# ── Core augmentation functions ────────────────────────────────────────────────

def apply_synonyms(question: str, n: int = 1) -> list[str]:
    """Return up to n variants of question by swapping one synonym per variant."""
    variants = set()
    for original, replacements in SYNONYMS.items():
        if original in question:
            for repl in replacements:
                new_q = question.replace(original, repl, 1)
                if new_q != question:
                    variants.add(new_q)
    return list(variants)[:n]


def apply_rewrite_frames(question: str) -> list[str]:
    """Apply prefix/suffix rewrite frames to the question."""
    # Strip trailing punctuation for frames that add their own
    q_stripped = question.rstrip("।?")
    variants = []
    for prefix, suffix in REWRITE_FRAMES:
        if prefix == "" and suffix == "":
            continue  # skip identity frame (base already exists)
        new_q = f"{prefix}{question}"
        if suffix and not new_q.endswith(suffix):
            new_q = f"{prefix}{q_stripped}{suffix}"
        if new_q != question:
            variants.append(new_q)
    return variants


def augment_template(template: dict) -> list[dict]:
    """Produce augmented variants of a single template."""
    base_q = template["bangla_question"]
    sql    = template["sql_query"]
    candidates = []

    # Synonym substitution
    for var_q in apply_synonyms(base_q, n=3):
        candidates.append(var_q)

    # Rewrite frames on base
    for var_q in apply_rewrite_frames(base_q):
        candidates.append(var_q)

    # Synonym substitution on already-rewritten variants (double augmentation)
    for rewritten in apply_rewrite_frames(base_q)[:2]:
        for var_q in apply_synonyms(rewritten, n=1):
            candidates.append(var_q)

    # Deduplicate within this template's candidates
    seen = {base_q}
    unique_candidates = []
    for q in candidates:
        q_stripped = q.strip()
        if q_stripped and q_stripped not in seen:
            seen.add(q_stripped)
            unique_candidates.append(q_stripped)

    # Cap
    selected = unique_candidates[:MAX_PARAPHRASES_PER_TEMPLATE]

    results = []
    for i, q in enumerate(selected, start=1):
        variant = deepcopy(template)
        variant["template_id"] = f"{template['template_id']}_aug{i:02d}"
        variant["bangla_question"] = q
        variant["is_augmented"] = True
        results.append(variant)

    return results


# ── Deduplication ──────────────────────────────────────────────────────────────

def deduplicate(pairs: list[dict]) -> list[dict]:
    """
    Remove exact-duplicate Bangla questions.
    Also removes pairs where a (question, sql) duplicate exists.
    """
    seen_questions = set()
    unique = []
    for pair in pairs:
        key = pair["bangla_question"].strip().lower()
        if key not in seen_questions:
            seen_questions.add(key)
            unique.append(pair)
    return unique


# ── Dataset split ──────────────────────────────────────────────────────────────

def split_dataset(all_pairs: list[dict]):
    """
    Split into train/dev/test.
    Strategy:
      - Within each difficulty tier (easy, medium), hold out 15% of unique
        query *patterns* (query_type) exclusively for test.
      - This ensures the test set always contains both easy and medium samples
        while still measuring generalization (not memorization of seen patterns).
    """
    def split_tier(pairs):
        by_pattern: dict[str, list[dict]] = {}
        for pair in pairs:
            qt = pair.get("query_type", "unknown")
            by_pattern.setdefault(qt, []).append(pair)

        patterns = list(by_pattern.keys())
        random.shuffle(patterns)

        n_test = max(1, round(len(patterns) * TEST_RATIO))
        n_dev  = max(1, round(len(patterns) * DEV_RATIO))

        test_patterns = set(patterns[:n_test])
        dev_patterns  = set(patterns[n_test:n_test + n_dev])

        tr, dv, te = [], [], []
        for qt, ps in by_pattern.items():
            random.shuffle(ps)
            if qt in test_patterns:
                te.extend(ps)
            elif qt in dev_patterns:
                dv.extend(ps)
            else:
                tr.extend(ps)
        return tr, dv, te

    easy_pairs   = [p for p in all_pairs if p["difficulty"] == "easy"]
    medium_pairs = [p for p in all_pairs if p["difficulty"] == "medium"]

    tr_e, dv_e, te_e = split_tier(easy_pairs)
    tr_m, dv_m, te_m = split_tier(medium_pairs)

    train = tr_e + tr_m
    dev   = dv_e + dv_m
    test  = te_e + te_m

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test



# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(templates, all_pairs, train, dev, test):
    easy_base   = sum(1 for t in templates if t["difficulty"] == "easy")
    medium_base = sum(1 for t in templates if t["difficulty"] == "medium")

    easy_aug   = sum(1 for p in all_pairs if p["difficulty"] == "easy")
    medium_aug = sum(1 for p in all_pairs if p["difficulty"] == "medium")

    return {
        "base_templates": {
            "easy":   easy_base,
            "medium": medium_base,
            "total":  len(templates),
        },
        "after_augmentation_dedup": {
            "easy":   easy_aug,
            "medium": medium_aug,
            "total":  len(all_pairs),
        },
        "splits": {
            "train": len(train),
            "dev":   len(dev),
            "test":  len(test),
        },
        "train_easy":   sum(1 for p in train if p["difficulty"] == "easy"),
        "train_medium": sum(1 for p in train if p["difficulty"] == "medium"),
        "dev_easy":     sum(1 for p in dev   if p["difficulty"] == "easy"),
        "dev_medium":   sum(1 for p in dev   if p["difficulty"] == "medium"),
        "test_easy":    sum(1 for p in test  if p["difficulty"] == "easy"),
        "test_medium":  sum(1 for p in test  if p["difficulty"] == "medium"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BanglaSQL Dataset Builder — Phase 2")
    print("=" * 60)

    # 1. Load base templates
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        templates = json.load(f)
    print(f"\n[1/5] Loaded {len(templates)} base templates")
    print(f"      Easy: {sum(1 for t in templates if t['difficulty']=='easy')}")
    print(f"      Medium: {sum(1 for t in templates if t['difficulty']=='medium')}")

    # 2. Mark base templates
    for t in templates:
        t["is_augmented"] = False

    # 3. Augment
    all_pairs = list(templates)  # start with base
    for template in templates:
        augmented = augment_template(template)
        all_pairs.extend(augmented)

    print(f"\n[2/5] After augmentation: {len(all_pairs)} pairs "
          f"(before deduplication)")

    # 4. Deduplicate
    all_pairs = deduplicate(all_pairs)
    print(f"\n[3/5] After deduplication: {len(all_pairs)} pairs")
    print(f"      Easy:   {sum(1 for p in all_pairs if p['difficulty']=='easy')}")
    print(f"      Medium: {sum(1 for p in all_pairs if p['difficulty']=='medium')}")

    # 5. Quality warning
    total = len(all_pairs)
    if total < 500:
        print(f"\n  [!] WARNING: Only {total} pairs. Quality > quantity for this "
              f"dataset size -- this is acceptable.")
    else:
        print(f"\n  [OK] Dataset size ({total} pairs) is within target range.")

    # 6. Split
    train, dev, test = split_dataset(all_pairs)
    print(f"\n[4/5] Split:")
    print(f"      Train : {len(train)} ({len(train)/total:.0%})")
    print(f"      Dev   : {len(dev)}   ({len(dev)/total:.0%})")
    print(f"      Test  : {len(test)}  ({len(test)/total:.0%})")
    print(f"      (Test patterns held out entirely from train for generalization)")

    # 7. Save
    os.makedirs(DATA_DIR, exist_ok=True)

    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        path = os.path.join(DATA_DIR, f"dataset_{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"\n[5/5] Saved {path}")

    stats = compute_stats(templates, all_pairs, train, dev, test)
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n      Saved stats: {stats_path}")
    print("\n" + "=" * 60)
    print("Dataset Stats Summary")
    print("=" * 60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("=" * 60)
    print("\nPhase 2 COMPLETE — dataset ready for Phase 3 model training.")


if __name__ == "__main__":
    main()
