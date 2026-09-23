"""
BanglaSQL — shared preprocessing, config resolution and SQL execution helpers.

train.py, evaluate.py and app.py all build model inputs through format_input and
resolve settings through load_config. A silent mismatch between the prompt used
at training time and the one used at inference is the classic way a working
checkpoint appears broken, so there is exactly one definition of each.
"""

import json
import os
import re
import sqlite3
import time
import unicodedata
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(BASE_DIR, "banglasql.db")

MODEL_CONFIG_NAME = "banglasql_config.json"
TASK_PREFIX = "translate Bangla to SQL:"

DEFAULT_SCHEMA_STRING = (
    "table: departments(dept_id, dept_name, building, phone) | "
    "table: instructors(instructor_id, first_name, last_name, email, dept_id, designation, joining_year) | "
    "table: students(student_id, first_name, last_name, email, dept_id, year_of_admission, cgpa) | "
    "table: courses(course_id, course_code, course_name, credits, dept_id, instructor_id, semester) | "
    "table: enrollments(enrollment_id, student_id, course_id, grade, grade_point) | "
    "table: attendance(attendance_id, student_id, course_id, date, status)"
)

DEFAULT_CONFIG = {
    "model_name":        "csebuetnlp/banglat5",
    "max_input_length":  256,
    "max_target_length": 128,
    "schema_string":     DEFAULT_SCHEMA_STRING,
    # Configs written before this key existed belong to checkpoints trained with
    # the schema appended, so the fallback must stay True for those to load right.
    "include_schema":    True,
}

MODEL_CONFIG_KEYS = ["model_name", "max_input_length", "max_target_length",
                     "schema_string", "include_schema"]


# ── Input formatting ───────────────────────────────────────────────────────────

def normalize_bangla(text: str) -> str:
    """NFC-normalize Bangla text so যুক্তাক্ষর have one representation."""
    return unicodedata.normalize("NFC", str(text).strip())


def format_input(question: str, config: dict) -> str:
    """Build the seq2seq source string: task prefix + question (+ schema if enabled)."""
    text = f"{TASK_PREFIX} {normalize_bangla(question)}"
    if config.get("include_schema"):
        text += f" schema: {config['schema_string']}"
    return text


def load_config(model_dir: str | None = None) -> dict:
    """Resolve input settings: defaults < data/train_config.json < the checkpoint's own copy.

    The checkpoint's copy wins so a model is always queried with the exact input
    format it was trained on, whatever data/train_config.json says today.
    """
    config = dict(DEFAULT_CONFIG)
    paths = [os.path.join(DATA_DIR, "train_config.json")]
    if model_dir:
        paths.append(os.path.join(model_dir, MODEL_CONFIG_NAME))
    for path in paths:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                config.update(json.load(f))
    return config


def save_model_config(config: dict, model_dir: str):
    """Store the input settings next to the weights they belong to."""
    with open(os.path.join(model_dir, MODEL_CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump({k: config[k] for k in MODEL_CONFIG_KEYS}, f, ensure_ascii=False, indent=2)


def load_split(split: str) -> list[dict]:
    """Load a dataset split, building the dataset first if it is missing."""
    path = os.path.join(DATA_DIR, f"dataset_{split}.json")
    if not os.path.exists(path):
        import build_dataset
        build_dataset.main()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── SQL execution ──────────────────────────────────────────────────────────────

def normalize_sql(sql: str) -> str:
    """Collapse whitespace, drop the trailing semicolon, lowercase."""
    sql = re.sub(r"\s+", " ", str(sql).strip())
    return sql.rstrip(";").strip().lower()


def open_readonly_db(path: str = DB_PATH) -> sqlite3.Connection:
    """Open the database read-only so a generated DROP/DELETE cannot do damage."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def run_sql(con: sqlite3.Connection, sql: str, timeout_s: float = 5.0):
    """Execute one SELECT, returning (rows, error). rows is None when execution failed."""
    if not normalize_sql(sql).startswith("select"):
        return None, "not a SELECT statement"
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return con.execute(sql).fetchall(), None
    except Exception as exc:
        return None, str(exc)
    finally:
        con.set_progress_handler(None, 0)


def results_match(gold_rows, pred_rows, gold_sql: str) -> bool:
    """Compare result sets; row order only matters when the gold query has ORDER BY."""
    if gold_rows is None or pred_rows is None:
        return False
    if len(gold_rows) != len(pred_rows):
        return False
    if "order by" in normalize_sql(gold_sql):
        return gold_rows == pred_rows
    return Counter(map(repr, gold_rows)) == Counter(map(repr, pred_rows))


def has_duplicate_select_columns(sql: str) -> bool:
    """True if the SELECT list names the same expression twice."""
    match = re.search(r"select\s+(?:distinct\s+)?(.*?)\s+from\s", normalize_sql(sql), re.S)
    if not match:
        return False
    columns = [c.strip() for c in re.split(r",(?![^()]*\))", match.group(1))]
    return len(columns) != len(set(columns))


def has_orphan_sort_direction(sql: str) -> bool:
    """True if ASC/DESC appears without an ORDER BY.

    `SELECT ... FROM students ASC LIMIT 3` executes, because SQLite reads `students
    ASC` as a table alias — so the query silently returns unordered rows.
    """
    norm = normalize_sql(sql)
    return bool(re.search(r"\b(asc|desc)\b", norm)) and "order by" not in norm


def is_decoding_artifact(sql: str) -> bool:
    """True if the query executes but is a recognisable generation defect.

    Both patterns run without error yet cannot be what the question asked for, so
    execution-guided decoding cannot filter them on execution alone. Neither appears
    in any of the 241 gold queries, so treating them as artifacts is safe.
    """
    return has_duplicate_select_columns(sql) or has_orphan_sort_direction(sql)


def tables_in(sql: str) -> frozenset:
    """The tables a query reads, from its FROM and JOIN clauses."""
    return frozenset(re.findall(r"(?:from|join) (\w+)", normalize_sql(sql)))


#: The Bangla word that names each column, for the column check in `over_projects`.
#: Hand-written, then verified: it vetoes none of the 2,170 gold queries in any split.
#: first_name and course_name are absent on purpose — both are just নাম in Bangla, so
#: a question can never distinguish them and neither can be checked this way.
COLUMN_CUES = {
    "credits":           ["ক্রেডিট"],
    "email":             ["ইমেইল", "মেইল"],
    "phone":             ["ফোন", "মোবাইল"],
    "building":          ["ভবন", "বিল্ডিং"],
    "designation":       ["পদবি", "পদবী", "পদ"],
    "joining_year":      ["যোগদান", "যোগ দেওয়া"],
    "year_of_admission": ["ভর্তি"],
    "semester":          ["সেমিস্টার"],
    "date":              ["তারিখ"],
    "status":            ["উপস্থিত", "অনুপস্থিত", "স্ট্যাটাস", "বিলম্ব"],
    "course_code":       ["কোড"],
    "grade_point":       ["পয়েন্ট"],
    "cgpa":              ["সিজিপিএ", "CGPA", "cgpa"],
}

_SELECT_LIST = re.compile(r"\s*SELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM", re.IGNORECASE | re.DOTALL)


def over_projects(question: str, sql: str) -> bool:
    """True if the SELECT list returns a column the question never asks for.

    The table head fixes *which tables* a query reads but not *which columns* it returns:
    asked for "কোর্সের নাম", a beam returning `course_name, credits` reads the right table
    and still answers a different question. This rejects it — `credits` is only returned
    when the question says ক্রেডিট.
    """
    match = _SELECT_LIST.match(sql)
    if not match:
        return False
    select_list = match.group(1)
    return any(re.search(rf"\b{column}\b", select_list) and not any(c in question for c in cues)
               for column, cues in COLUMN_CUES.items())


def pick_executable(candidates: list[str], con: sqlite3.Connection,
                    preferred_tables: frozenset | None = None,
                    question: str | None = None) -> str:
    """Execution-guided decoding (Wang et al., 2018): first beam that runs without error.

    Beams arrive in model-score order, so the model's ranking is kept among valid
    queries and only overridden when a higher-scoring query cannot execute or is a
    recognisable artifact. Preference order: executes and is not an artifact, then
    merely executes, then the top beam.

    Two optional filters narrow the choice further, each falling back when it would leave
    nothing, so neither can ever leave a question without an answer:

    `question`          the column check — beams returning a column the question never
                        mentions are skipped (over_projects)
    `preferred_tables`  the table head's prediction (table_head.predict_tables) — the first
                        remaining beam reading exactly those tables wins, which is what
                        lets a `wrong_table` beam, one that runs and so passes every other
                        check, be skipped
    """
    executable = [sql for sql in candidates if run_sql(con, sql)[1] is None]
    clean = [sql for sql in executable if not is_decoding_artifact(sql)]
    if question is not None:
        clean = [sql for sql in clean if not over_projects(question, sql)] or clean
    if preferred_tables:
        for sql in clean:
            if tables_in(sql) == preferred_tables:
                return sql
    if clean:
        return clean[0]
    return executable[0] if executable else candidates[0]
