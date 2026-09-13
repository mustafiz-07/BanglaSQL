"""
BanglaSQL — Phase 5: Demo Interface

Bangla question in → generated SQL → live result table from the database.

Run:
    streamlit run app.py

Safety: the database is opened read-only, generated SQL must be a single SELECT
over known tables, and generation is bounded by a token limit and beam count.
"""

import re
import time

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from common import format_input, load_config, open_readonly_db, run_sql

MODEL_DIR = "checkpoints/best_model"
NUM_BEAMS = 4

EXAMPLE_QUESTIONS = [
    "সকল শিক্ষার্থীর তালিকা দাও।",
    "যেসব শিক্ষার্থীর CGPA ৩.৫-এর বেশি তাদের নাম দাও।",
    "প্রতিটি বিভাগে কতজন শিক্ষার্থী আছে?",
    "গণিত বিভাগের শিক্ষকদের নাম দাও।",
    "সবচেয়ে বেশি CGPA কত?",
]


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_schema_tables() -> dict[str, list[str]]:
    con = open_readonly_db()
    tables = {}
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        tables[name] = [row[1] for row in con.execute(f"PRAGMA table_info({name})")]
    con.close()
    return tables


def validate_sql(sql: str, known_tables: dict) -> str | None:
    """Return a Bangla error message if the SQL is unsafe or references unknown tables."""
    normalized = re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()

    if not normalized.startswith("select"):
        return "শুধুমাত্র SELECT কোয়েরি চালানো যাবে।"
    if ";" in normalized:
        return "একাধিক স্টেটমেন্ট চালানো যাবে না।"

    referenced = set(re.findall(r"(?:from|join)\s+(\w+)", normalized))
    unknown = referenced - {t.lower() for t in known_tables}
    if unknown:
        return f"অজানা টেবিল: {', '.join(sorted(unknown))}"
    return None


def generate_candidates(tokenizer, model, question: str, config: dict) -> list[str]:
    enc = tokenizer(
        format_input(question, config),
        max_length=int(config["max_input_length"]),
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=int(config["max_target_length"]),
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS,
            early_stopping=True,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def choose_sql(candidates: list[str], tables: dict) -> tuple[str, int]:
    """Execution-guided choice: first beam that passes validation and executes."""
    con = open_readonly_db()
    try:
        for rank, sql in enumerate(candidates):
            if validate_sql(sql, tables) is None and run_sql(con, sql)[1] is None:
                return sql, rank
    finally:
        con.close()
    return candidates[0], 0


def run_query(sql: str) -> pd.DataFrame:
    con = open_readonly_db()
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def main():
    st.set_page_config(page_title="BanglaSQL", page_icon="🗄️", layout="wide")
    st.title("BanglaSQL")
    st.caption("বাংলা প্রশ্ন থেকে SQL কোয়েরি — University Management System")

    config = load_config(MODEL_DIR)
    tables = load_schema_tables()

    with st.sidebar:
        st.subheader("ডাটাবেস স্কিমা")
        for table, columns in tables.items():
            if table.startswith("sqlite_"):
                continue
            with st.expander(table):
                st.write(", ".join(columns))

    example = st.selectbox("উদাহরণ প্রশ্ন", [""] + EXAMPLE_QUESTIONS)
    question = st.text_input("আপনার প্রশ্ন লিখুন (বাংলায়)", value=example)

    if not st.button("SQL তৈরি করো", type="primary") or not question.strip():
        return

    try:
        tokenizer, model = load_model()
    except Exception:
        st.error(f"মডেল লোড করা যায়নি। `{MODEL_DIR}` আছে কিনা দেখুন (আগে train.py চালান)।")
        return

    start = time.time()
    with st.spinner("SQL তৈরি হচ্ছে..."):
        candidates = generate_candidates(tokenizer, model, question, config)
        sql, rank = choose_sql(candidates, tables)
    st.code(sql, language="sql")
    note = f"জেনারেশন সময়: {time.time() - start:.2f} সেকেন্ড"
    if rank > 0:
        note += f" · শীর্ষ কোয়েরিটি চালানো যায়নি, তাই {rank + 1} নম্বর বিকল্পটি নেওয়া হয়েছে"
    st.caption(note)

    error = validate_sql(sql, tables)
    if error:
        st.error(f"কোয়েরিটি চালানো গেল না — {error}")
        return

    try:
        df = run_query(sql)
    except Exception as exc:
        st.error("তৈরি হওয়া SQL কোয়েরিটি সঠিক নয়, তাই ফলাফল দেখানো যাচ্ছে না।")
        st.caption(f"SQLite: {exc}")
        return

    if df.empty:
        st.warning("এই প্রশ্নের জন্য ডাটাবেসে কোনো ফলাফল পাওয়া যায়নি।")
    else:
        st.success(f"{len(df)} টি ফলাফল পাওয়া গেছে")
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
