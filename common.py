"""
BanglaSQL — shared preprocessing and config loading.

train.py, evaluate.py and app.py all import format_input from here. A silent
mismatch between the prompt used at training time and the one used at inference
is the classic way a working checkpoint appears broken, so there is exactly one
definition of it.
"""

import json
import os
import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(BASE_DIR, "banglasql.db")

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
}


def normalize_bangla(text: str) -> str:
    """NFC-normalize Bangla text so যুক্তাক্ষর have one representation."""
    return unicodedata.normalize("NFC", str(text).strip())


def format_input(question: str, schema: str) -> str:
    """Build the seq2seq source string: task prefix + question + linearized schema."""
    return f"{TASK_PREFIX} {normalize_bangla(question)} schema: {schema}"


def load_config() -> dict:
    """Load data/train_config.json, falling back to defaults for missing keys."""
    path = os.path.join(DATA_DIR, "train_config.json")
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            config.update(json.load(f))
    return config


def load_split(split: str) -> list[dict]:
    """Load a dataset split, building the dataset first if it is missing."""
    path = os.path.join(DATA_DIR, f"dataset_{split}.json")
    if not os.path.exists(path):
        import build_dataset
        build_dataset.main()
    with open(path, encoding="utf-8") as f:
        return json.load(f)
