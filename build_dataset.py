"""
BanglaSQL — Dataset Builder (Phase 2)

Loads base templates, generates value-slot variants, augments via rule-based
Bangla paraphrasing, deduplicates, and produces train/dev/test splits.

Two kinds of augmentation, doing different jobs:
  - Value-slot variants swap one literal (department, CGPA/year threshold, grade,
    semester, course, LIMIT) consistently in the question and the SQL. They add
    new SQL *targets*. Without them the 107 training templates gave the model
    just 107 distinct SQL strings to recall; the first run memorised them (train
    loss 0.01, dev loss rising from epoch 3) and 49 of 66 invalid test queries
    were schema-grounding errors such as `students WHERE grade = ...`.
  - Paraphrases (synonyms + register frames) vary only the Bangla wording.

Split strategy — template-level holdout, stratified by query_type:
  Every variant (value or paraphrase) stays with its base template, so nothing
  derived from a training template can reach dev/test. Templates are split
  within each query_type, so train sees every SQL shape while dev/test consist
  entirely of unseen templates.

Run (after create_database.py — value variants are checked against the DB):
    python build_dataset.py

Outputs:
    data/dataset_train.json
    data/dataset_dev.json
    data/dataset_test.json
    data/dataset_stats.json
"""

import json
import os
import random
import re
import sqlite3
from collections import defaultdict
from copy import deepcopy

from common import DB_PATH

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEMPLATES_PATH = os.path.join(DATA_DIR, "templates.json")

MAX_PARAPHRASES_PER_TEMPLATE  = 5
VALUE_VARIANTS_PER_TEMPLATE   = 3
PARAPHRASES_PER_VALUE_VARIANT = 2
TRAIN_RATIO = 0.70
DEV_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Synonym dictionary (standard Cholito Bangla) ──────────────────────────────
# Every entry must be meaning-preserving and grammatical in place. Substitutions
# that shift meaning (গ্রেড→মার্ক), change register to Sadhu (ভর্তি হইয়াছে), or
# require the rest of the clause to be rewritten (বেশি→ঊর্ধ্বে, which needs a
# different postposition) are deliberately excluded.

SYNONYMS = {
    # request verbs
    "দাও।":        ["দেখাও।", "জানাও।", "প্রদর্শন করো।"],
    "দেখাও।":      ["দাও।", "জানাও।", "প্রদর্শন করো।"],
    "বলো।":        ["দাও।", "দেখাও।", "জানাও।"],
    "হিসাব করো।":  ["বের করো।", "নির্ণয় করো।"],
    "বের করো।":    ["হিসাব করো।", "নির্ণয় করো।"],

    # query subjects
    "তালিকা দাও":  ["তালিকা দেখাও", "তথ্য দাও"],
    "তথ্য দাও":    ["তালিকা দাও", "বিস্তারিত দাও"],
    "নাম দাও":     ["নাম দেখাও", "নামের তালিকা দাও"],

    # quantifiers
    "সকল":   ["সব", "সমস্ত"],
    "সব":    ["সকল", "সমস্ত"],
    "সমস্ত": ["সকল", "সব"],
    "যেসব":  ["যারা", "যে সকল"],
    "যারা":  ["যেসব", "যে সকল"],
    "কারা":  ["কে কে", "কোন কোন"],
    "প্রতিটি": ["প্রত্যেক", "প্রতিটা"],

    # comparison
    "বেশি": ["অধিক", "বেশী"],

    # ordering
    "অবরোহী ক্রমে": ["বড় থেকে ছোট ক্রমে", "উচ্চ থেকে নিম্ন ক্রমে"],
    "আরোহী ক্রমে":  ["ছোট থেকে বড় ক্রমে", "নিম্ন থেকে উচ্চ ক্রমে"],
    "অনুযায়ী সাজিয়ে": ["অনুসারে সাজিয়ে", "অনুযায়ী ক্রমে"],

    # admission / joining
    "ভর্তি হয়েছে":   ["ভর্তি হয়েছিল", "ভর্তি নিয়েছে"],
    "যোগ দিয়েছেন":  ["যোগদান করেছেন", "নিয়োগ পেয়েছেন"],

    # academic vocabulary
    "নথিভুক্ত":       ["ভর্তি", "এনরোল"],
    "শিক্ষার্থী":     ["ছাত্রছাত্রী", "স্টুডেন্ট"],
    "শিক্ষার্থীর":    ["ছাত্রছাত্রীর", "স্টুডেন্টের"],
    "শিক্ষার্থীদের":  ["ছাত্রছাত্রীদের", "স্টুডেন্টদের"],
    "শিক্ষক":        ["ইন্সট্রাক্টর"],
    "শিক্ষকের":      ["ইন্সট্রাক্টরের"],
    "শিক্ষকদের":     ["ইন্সট্রাক্টরদের"],
    "বিভাগ":   ["ডিপার্টমেন্ট"],
    "বিভাগের": ["ডিপার্টমেন্টের"],
    "বিভাগে":  ["ডিপার্টমেন্টে"],
    "কোর্স":   ["বিষয়"],
    "কোর্সে":  ["বিষয়ে"],
    "কোর্সের": ["বিষয়ের"],
    "উপস্থিত":   ["হাজির", "প্রেজেন্ট"],
    "অনুপস্থিত": ["গরহাজির", "অ্যাবসেন্ট"],

    # aggregates
    "গড়":      ["গড় মান", "এভারেজ"],
    "মোট":     ["সর্বমোট", "টোটাল"],
    "সর্বোচ্চ": ["সবচেয়ে বেশি", "সর্বাধিক"],
    "সর্বনিম্ন": ["সবচেয়ে কম", "ন্যূনতম"],
    "সংখ্যা":   ["পরিমাণ"],

    # question starters
    "কতজন": ["কত সংখ্যক", "মোট কতজন"],
    "কতটি": ["কতগুলো", "কয়টি"],
}

# ── Register frames ────────────────────────────────────────────────────────────
# Imperative politeness markers (অনুগ্রহ করে / দয়া করে) only fit commands ending
# in "।"; prefixing them onto a "কত?" question is ungrammatical.

COMMAND_FRAMES  = ["অনুগ্রহ করে, ", "দয়া করে ", "ডাটাবেস থেকে "]
QUESTION_FRAMES = ["আমি জানতে চাই, ", "বলো তো, ", "একটু বলো, "]

# ── Value slots ────────────────────────────────────────────────────────────────
# Only literals whose Bangla surface form can be located unambiguously are used.
# Attendance status and designations are left out on purpose: অনুপস্থিত contains
# উপস্থিত and সহকারী অধ্যাপক contains অধ্যাপক, so a swap could hit the wrong word.

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

DEPARTMENTS = {  # SQL value -> Bangla surface forms, the one used for new questions first
    "Computer Science and Engineering":      ["কম্পিউটার সায়েন্স"],
    "Electrical and Electronic Engineering": ["EEE", "ইইই"],
    "Mathematics":             ["গণিত"],
    "Physics":                 ["পদার্থবিজ্ঞান"],
    "Chemistry":               ["রসায়ন"],
    "English":                 ["ইংরেজি"],
    "Economics":               ["অর্থনীতি"],
    "Business Administration": ["ব্যবসায় প্রশাসন"],
    "Civil Engineering":       ["সিভিল ইঞ্জিনিয়ারিং"],
    "Mechanical Engineering":  ["মেকানিক্যাল ইঞ্জিনিয়ারিং"],
}

NUMERIC_POOLS = {  # SQL column (or alias) -> plausible thresholds, as written in SQL
    "cgpa":              ["2.5", "3.0", "3.25", "3.5", "3.75"],
    "avg_cgpa":          ["2.75", "3.0", "3.25"],
    "grade_point":       ["2.0", "2.5", "3.0", "3.5"],
    "avg_gp":            ["2.5", "3.0", "3.5"],
    "year_of_admission": ["2020", "2021", "2022", "2023"],
    "joining_year":      ["2005", "2010", "2015", "2018"],
}
EQUALITY_COLUMNS = {"year_of_admission", "joining_year"}
LIMIT_POOL = ["3", "5", "10"]

LATIN_POOLS = {  # literals written in Latin script inside the Bangla question
    "grade":       ["A+", "A", "B+", "B", "F"],
    "semester":    ["Spring 2023", "Fall 2023", "Spring 2024", "Summer 2024", "Fall 2024"],
    "course_name": ["Data Structures", "Algorithms", "Database Systems", "Operating Systems",
                    "Computer Networks", "Machine Learning", "Digital Electronics", "Thermodynamics"],
}


# ── Paraphrase augmentation ────────────────────────────────────────────────────

def apply_synonyms(question: str, n: int = 3) -> list[str]:
    """Return up to n variants, each swapping one synonym.

    Candidates are collected in an order-preserving list, not a set: Python
    randomizes string hashing per process, so slicing list(a_set)[:n] returns a
    different subset on every run even with random.seed() fixed.
    """
    variants, seen = [], set()
    for original, replacements in SYNONYMS.items():
        if original not in question:
            continue
        for repl in replacements:
            new_q = question.replace(original, repl, 1)
            if new_q != question and new_q not in seen:
                seen.add(new_q)
                variants.append(new_q)
    return variants[:n]


def apply_frames(question: str) -> list[str]:
    """Prepend a register marker appropriate to the sentence type."""
    frames = QUESTION_FRAMES if question.rstrip().endswith("?") else COMMAND_FRAMES
    return [f"{frame}{question}" for frame in frames]


def augment_template(template: dict, max_variants: int) -> list[dict]:
    """Produce up to max_variants paraphrases of one question–SQL pair."""
    base_q = template["bangla_question"]

    # Synonyms first: they vary the content words the model must ground onto
    # schema elements. Frames only vary the wrapper.
    candidates = apply_synonyms(base_q, n=3) + apply_frames(base_q)

    seen = {base_q}
    unique = []
    for q in candidates:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)

    results = []
    for i, q in enumerate(unique[:max_variants], start=1):
        variant = deepcopy(template)
        variant["template_id"] = f"{template['template_id']}_aug{i:02d}"
        variant["bangla_question"] = q
        variant["is_augmented"] = True
        results.append(variant)
    return results


# ── Value-slot augmentation ────────────────────────────────────────────────────

def _single_match(pattern: str, text: str):
    matches = list(re.finditer(pattern, text))
    return matches[0] if len(matches) == 1 else None


def find_value_slots(question: str, sql: str) -> list[tuple]:
    """Locate literals present in both the SQL and, recognisably, the question.

    Returns (sql_span, question_span, old_value, pool, render) tuples; render maps
    a pool value to its surface form in the question. A literal is used only when
    it occurs exactly once on each side, so a swap can never touch another token.
    """
    # Blank out quoted strings (same length) so numbers inside them are ignored.
    masked = re.sub(r"'[^']*'", lambda m: "'" + "_" * (len(m.group()) - 2) + "'", sql)
    slots = []

    def numeric_slot(value, sql_span, pool):
        if len(re.findall(rf"(?<![\w.]){re.escape(value)}(?![\w.])", masked)) != 1:
            return
        bn = re.escape(value.translate(BN_DIGITS))
        qm = _single_match(rf"(?<![০-৯.]){bn}(?!\.?[০-৯])", question)
        if qm:
            slots.append((sql_span, qm.span(), value, pool, lambda v: v.translate(BN_DIGITS)))

    for m in re.finditer(r"(?:\b\w+\.)?(\w+)\s*(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)\b", masked):
        column, op, value = m.group(1).lower(), m.group(2), m.group(3)
        pool = NUMERIC_POOLS.get(column)
        if pool and (op != "=" or column in EQUALITY_COLUMNS):
            numeric_slot(value, m.span(3), pool)

    for m in re.finditer(r"\bLIMIT\s+(\d+)\b", masked, re.IGNORECASE):
        if m.group(1) != "1":  # LIMIT 1 comes from সবচেয়ে/সর্বোচ্চ, not a number in the question
            numeric_slot(m.group(1), m.span(1), LIMIT_POOL)

    for m in re.finditer(r"(?:\b\w+\.)?(\w+)\s*=\s*'([^']*)'", sql):
        column, value = m.group(1).lower(), m.group(2)
        if sql.count(f"'{value}'") != 1:
            continue
        if column == "dept_name" and value in DEPARTMENTS:
            for form in DEPARTMENTS[value]:
                qm = _single_match(rf"(?<![ঀ-৿A-Za-z]){re.escape(form)}", question)
                if qm:
                    slots.append((m.span(2), qm.span(), value, list(DEPARTMENTS),
                                  lambda v: DEPARTMENTS[v][0]))
                    break
        elif column in LATIN_POOLS:
            qm = _single_match(rf"(?<![A-Za-z0-9+\-]){re.escape(value)}(?![A-Za-z0-9+\-])", question)
            if qm:
                slots.append((m.span(2), qm.span(), value, LATIN_POOLS[column], lambda v: v))

    return slots


def value_variants(template: dict, known_sql: set, con: sqlite3.Connection) -> list[dict]:
    """New question–SQL pairs from one template, each with a single literal swapped.

    A variant is kept only if its SQL executes and returns rows, and its SQL is not
    already present anywhere (base templates or earlier variants) — the latter keeps
    a train variant from reproducing a held-out template's query.
    """
    question, sql = template["bangla_question"], template["sql_query"]
    candidates = [
        (sql_span, q_span, new, render)
        for sql_span, q_span, old, pool, render in find_value_slots(question, sql)
        for new in pool
        if new != old
    ]
    random.shuffle(candidates)

    variants = []
    for sql_span, q_span, new, render in candidates:
        if len(variants) >= VALUE_VARIANTS_PER_TEMPLATE:
            break
        new_sql = sql[:sql_span[0]] + new + sql[sql_span[1]:]
        if new_sql in known_sql:
            continue
        try:
            if not con.execute(new_sql).fetchall():
                continue
        except sqlite3.Error:
            continue
        known_sql.add(new_sql)

        variant = deepcopy(template)
        variant["template_id"] = f"{template['template_id']}_v{len(variants) + 1}"
        variant["bangla_question"] = question[:q_span[0]] + render(new) + question[q_span[1]:]
        variant["sql_query"] = new_sql
        variant["is_augmented"] = True
        variant["is_value_variant"] = True
        variants.append(variant)
    return variants


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove exact-duplicate Bangla questions, keeping first occurrence."""
    seen, unique = set(), []
    for pair in pairs:
        key = pair["bangla_question"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


# ── Dataset split ──────────────────────────────────────────────────────────────

def split_dataset(all_pairs: list[dict]):
    """Split by base template, stratified by query_type.

    Within each query_type the *templates* are partitioned 70/15/15, then every
    variant follows its base template into that split. Train therefore contains
    every query_type, and dev/test templates are entirely unseen.

    Query types with a single template go wholly to train — holding out the only
    example of a shape would make it unlearnable.
    """
    by_template = defaultdict(list)
    for pair in all_pairs:
        by_template[pair["base_template_id"]].append(pair)

    templates_by_type = defaultdict(list)
    for tid, pairs in by_template.items():
        templates_by_type[pairs[0].get("query_type", "unknown")].append(tid)

    train_t, dev_t, test_t = [], [], []
    for qtype in sorted(templates_by_type):
        tids = sorted(templates_by_type[qtype])
        random.shuffle(tids)
        n = len(tids)

        if n == 1:
            train_t += tids
        elif n == 2:
            train_t.append(tids[0])
            test_t.append(tids[1])
        else:
            n_test = max(1, round(n * TEST_RATIO))
            n_dev  = max(1, round(n * DEV_RATIO))
            while n_test + n_dev >= n:           # always leave train >= 1
                if n_dev > 1:
                    n_dev -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            test_t  += tids[:n_test]
            dev_t   += tids[n_test:n_test + n_dev]
            train_t += tids[n_test + n_dev:]

    def collect(tids):
        out = []
        for tid in tids:
            out.extend(by_template[tid])
        random.shuffle(out)
        return out

    return collect(train_t), collect(dev_t), collect(test_t)


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(templates, value_pairs, all_pairs, train, dev, test):
    def tier(pairs, level):
        return sum(1 for p in pairs if p["difficulty"] == level)

    def unique_sql(pairs):
        return len({p["sql_query"] for p in pairs})

    train_types = {p["query_type"] for p in train}
    return {
        "base_templates": {
            "easy":   tier(templates, "easy"),
            "medium": tier(templates, "medium"),
            "total":  len(templates),
        },
        "value_slot_variants": len(value_pairs),
        "after_augmentation_dedup": {
            "easy":   tier(all_pairs, "easy"),
            "medium": tier(all_pairs, "medium"),
            "total":  len(all_pairs),
            "augmentation_ratio": round(len(all_pairs) / len(templates), 2),
        },
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "templates_per_split": {
            "train": len({p["base_template_id"] for p in train}),
            "dev":   len({p["base_template_id"] for p in dev}),
            "test":  len({p["base_template_id"] for p in test}),
        },
        "unique_sql_per_split": {
            "train": unique_sql(train),
            "dev":   unique_sql(dev),
            "test":  unique_sql(test),
        },
        "train_easy":   tier(train, "easy"),
        "train_medium": tier(train, "medium"),
        "dev_easy":     tier(dev, "easy"),
        "dev_medium":   tier(dev, "medium"),
        "test_easy":    tier(test, "easy"),
        "test_medium":  tier(test, "medium"),
        "query_types": {
            "total":            len({p["query_type"] for p in all_pairs}),
            "in_train":         len(train_types),
            "dev_unseen_in_train":  sorted({p["query_type"] for p in dev}  - train_types),
            "test_unseen_in_train": sorted({p["query_type"] for p in test} - train_types),
        },
        "sql_leakage": {
            "dev_sql_also_in_train":  len({p["sql_query"] for p in dev}  & {p["sql_query"] for p in train}),
            "test_sql_also_in_train": len({p["sql_query"] for p in test} & {p["sql_query"] for p in train}),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BanglaSQL Dataset Builder — Phase 2")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        raise SystemExit(f"Database not found: {DB_PATH}\nRun: python create_database.py")

    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        templates = json.load(f)
    print(f"\n[1/5] Loaded {len(templates)} base templates")
    print(f"      Easy  : {sum(1 for t in templates if t['difficulty'] == 'easy')}")
    print(f"      Medium: {sum(1 for t in templates if t['difficulty'] == 'medium')}")

    for t in templates:
        t["is_augmented"] = False
        t["base_template_id"] = t["template_id"]

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    known_sql = {t["sql_query"] for t in templates}
    value_pairs = []
    for template in templates:
        value_pairs.extend(value_variants(template, known_sql, con))
    con.close()
    print(f"\n[2/5] Value-slot variants: {len(value_pairs)} new question–SQL pairs")

    all_pairs = list(templates) + value_pairs
    for template in templates:
        all_pairs.extend(augment_template(template, MAX_PARAPHRASES_PER_TEMPLATE))
    for variant in value_pairs:
        all_pairs.extend(augment_template(variant, PARAPHRASES_PER_VALUE_VARIANT))

    all_pairs = deduplicate(all_pairs)
    print(f"\n[3/5] After paraphrasing + dedup: {len(all_pairs)} pairs "
          f"({len(all_pairs) / len(templates):.1f}x base)")

    train, dev, test = split_dataset(all_pairs)
    total = len(all_pairs)
    print(f"\n[4/5] Split (by template, stratified by query_type):")
    print(f"      Train : {len(train):4d} ({len(train)/total:.0%})")
    print(f"      Dev   : {len(dev):4d} ({len(dev)/total:.0%})")
    print(f"      Test  : {len(test):4d} ({len(test)/total:.0%})")

    stats = compute_stats(templates, value_pairs, all_pairs, train, dev, test)
    unseen_dev  = stats["query_types"]["dev_unseen_in_train"]
    unseen_test = stats["query_types"]["test_unseen_in_train"]
    if unseen_dev or unseen_test:
        print(f"\n  [!] Query types missing from train — dev: {unseen_dev}, test: {unseen_test}")
    else:
        print(f"\n  [OK] All {stats['query_types']['total']} query types present in train.")
    print(f"  [OK] Distinct SQL targets in train: {stats['unique_sql_per_split']['train']}")
    print(f"  [OK] Test SQL queries also seen in train: "
          f"{stats['sql_leakage']['test_sql_also_in_train']} (0 = no leakage)")

    os.makedirs(DATA_DIR, exist_ok=True)
    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        path = os.path.join(DATA_DIR, f"dataset_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n[5/5] Saved {path}")

    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n      Saved stats: {stats_path}")

    print("\n" + "=" * 60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("=" * 60)
    print("\nPhase 2 COMPLETE — dataset ready for Phase 3 model training.")


if __name__ == "__main__":
    main()
