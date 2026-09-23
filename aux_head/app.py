"""
BanglaSQL — Phase 5: Demo Interface

Bangla question in → generated SQL → live result table from the database.

Run:
    streamlit run app.py

    # or against a checkpoint somewhere else:
    BANGLASQL_MODEL="best_model (2)" streamlit run app.py

The sidebar exposes the inference-time work, so the decoding strategies can be compared
on the same question rather than only read about:

    execution-guided        generate 4 beams, return the highest-scoring one that runs
                            (Wang et al., 2018)
    + constrained fallback  re-generate with the schema constraint only when no beam
                            runs. Validity 96.0% → 98.2% at no cost to accuracy.

Constraining *every* question is deliberately not offered: it reaches the same validity
and loses ~2.8 points of execution accuracy, because execution-guided selection had been
using invalidity as a signal and the constraint removes it.

Safety: the database is opened read-only, generated SQL must be a single SELECT over
known tables, and generation is bounded by a token limit and beam count.
"""

import os
import re
import time

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from common import (
    format_input, is_decoding_artifact, load_config, open_readonly_db, over_projects, run_sql,
    tables_in,
)
from table_head import load_head, predict_tables

# Point at a checkpoint elsewhere without editing the file — the trained model is often
# unzipped beside the repo rather than into checkpoints/.
MODEL_DIR = os.environ.get("BANGLASQL_MODEL", "checkpoints/best_model")
NUM_BEAMS = 4

BASELINE = "execution-guided"
FALLBACK = "+ schema-constrained fallback"

EXAMPLE_QUESTIONS = [
    "সকল শিক্ষার্থীর তালিকা দাও।",
    "যেসব শিক্ষার্থীর CGPA ৩.৫-এর বেশি তাদের নাম দাও।",
    "প্রতিটি বিভাগে কতজন শিক্ষার্থী আছে?",
    "গণিত বিভাগের শিক্ষকদের নাম দাও।",
    "সবচেয়ে বেশি CGPA কত?",
    "গড় CGPA কত?",
    "প্রতিটি কোর্সের নাম ও ক্রেডিট দেখাও।",
    "কোন কোর্সে সবচেয়ে বেশি শিক্ষার্থী নথিভুক্ত?",
]


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_table_head():
    """The auxiliary table head saved with the checkpoint, or None if it has none."""
    return load_head(MODEL_DIR)


@st.cache_resource
def load_decoder(max_length: int):
    """The schema-constrained decoder, or None if the grammar cannot be built.

    Cached separately from the model because building it scans the whole vocabulary.
    A failure here disables the constrained modes rather than taking the app down —
    the baseline decoder does not depend on it.
    """
    try:
        from constrained_decode import ConstrainedDecoder
        tokenizer, _ = load_model()
        return ConstrainedDecoder(tokenizer, max_length=max_length)
    except Exception as exc:  # noqa: BLE001 - surfaced in the sidebar, not swallowed
        st.session_state["decoder_error"] = str(exc)
        return None


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


def generate_candidates(tokenizer, model, question: str, config: dict,
                        decoder=None) -> list[str]:
    enc = tokenizer(
        format_input(question, config),
        max_length=int(config["max_input_length"]),
        truncation=True,
        return_tensors="pt",
    )
    gen_kwargs = {
        "max_length": int(config["max_target_length"]),
        "num_beams": NUM_BEAMS,
        "num_return_sequences": NUM_BEAMS,
        "early_stopping": True,
    }
    if decoder is not None:
        # Masks the distribution at every step so only schema-valid continuations are
        # reachable; see constrained_decode.ConstrainedDecoder.
        gen_kwargs["prefix_allowed_tokens_fn"] = decoder.prefix_fn()
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def executable_ranks(candidates: list[str], tables: dict) -> list[int]:
    """Indices of the candidates that validate and execute, in model-score order."""
    con = open_readonly_db()
    try:
        return [rank for rank, sql in enumerate(candidates)
                if validate_sql(sql, tables) is None and run_sql(con, sql)[1] is None]
    finally:
        con.close()


def choose_sql(candidates: list[str], tables: dict,
               preferred_tables: frozenset | None = None,
               question: str | None = None) -> tuple[str, int]:
    """Execution-guided choice (Wang et al., 2018): the best beam that actually runs.

    Beams arrive in model-score order, so the model's own ranking is kept among queries
    that execute, and is overridden only when a higher-scoring one fails to run or is a
    recognisable decoding artifact. Returns (sql, rank) so the UI can say when a lower
    beam was used.

    `question` enables the column check (beams returning a column the question never
    mentions are skipped); `preferred_tables` is the table head's prediction, and the first
    remaining beam reading exactly those tables wins. Same rules, in the same order, as
    common.pick_executable, so the app and evaluate.py choose identically.
    """
    usable = executable_ranks(candidates, tables)
    clean = [r for r in usable if not is_decoding_artifact(candidates[r])]
    if question is not None:
        clean = [r for r in clean if not over_projects(question, candidates[r])] or clean
    if preferred_tables:
        for r in clean:
            if tables_in(candidates[r]) == preferred_tables:
                return candidates[r], r
    if clean:
        return candidates[clean[0]], clean[0]
    if usable:
        return candidates[usable[0]], usable[0]
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
        st.subheader("ডিকোডিং কৌশল")
        mode = st.radio(
            "Decoding strategy",
            [BASELINE, FALLBACK],
            index=1,
            help="Constrained decoding masks the vocabulary at every step so only "
                 "schema-valid continuations are reachable. Fallback applies it only "
                 "when no beam executes, which keeps execution-guided reranking intact.",
        )
        show_candidates = st.checkbox("show all candidate queries", value=False)

        st.divider()
        st.subheader("ডাটাবেস স্কিমা")
        for table, columns in tables.items():
            if table.startswith("sqlite_"):
                continue
            with st.expander(table):
                st.write(", ".join(columns))

        st.divider()
        st.caption(f"Checkpoint: `{MODEL_DIR}`")

    example = st.selectbox("উদাহরণ প্রশ্ন", [""] + EXAMPLE_QUESTIONS)
    question = st.text_input("আপনার প্রশ্ন লিখুন (বাংলায়)", value=example)

    if not st.button("SQL তৈরি করো", type="primary") or not question.strip():
        return

    try:
        tokenizer, model = load_model()
    except Exception:
        st.error(f"মডেল লোড করা যায়নি। `{MODEL_DIR}` আছে কিনা দেখুন (আগে train.py চালান)।")
        return

    decoder = None
    if mode == FALLBACK:
        decoder = load_decoder(int(config["max_target_length"]))
        if decoder is None:
            st.warning("Schema constraint could not be built; falling back to "
                       f"execution-guided decoding. {st.session_state.get('decoder_error', '')}")

    start = time.time()
    notes = []
    with st.spinner("SQL তৈরি হচ্ছে..."):
        candidates = generate_candidates(tokenizer, model, question, config)

        # Fallback mode constrains only when reranking has nothing to fall through to.
        # That is the whole point: the constraint repairs invalid queries, but applying it
        # everywhere also makes *wrong* queries valid and so defeats the reranker.
        if mode == FALLBACK and decoder is not None and not executable_ranks(candidates, tables):
            notes.append("কোনো বিকল্প চালানো যায়নি — স্কিমা-কনস্ট্রেইন্ট দিয়ে আবার তৈরি করা হয়েছে")
            candidates = generate_candidates(tokenizer, model, question, config,
                                             decoder=decoder)

        head = load_table_head()
        predicted = (predict_tables(model, head, tokenizer, [question], config)[0]
                     if head is not None else None)
        sql, rank = choose_sql(candidates, tables, predicted, question)

    st.code(sql, language="sql")

    notes.insert(0, f"জেনারেশন সময়: {time.time() - start:.2f} সেকেন্ড")
    if predicted is not None:
        notes.append("প্রত্যাশিত টেবিল: " + (", ".join(sorted(predicted)) or "—"))
    if rank > 0:
        notes.append(f"{rank + 1} নম্বর বিকল্পটি নেওয়া হয়েছে")
    st.caption(" · ".join(notes))

    if show_candidates:
        with st.expander(f"সব বিকল্প ({len(candidates)}টি, মডেল-স্কোর অনুসারে)"):
            runs = set(executable_ranks(candidates, tables))
            for i, candidate in enumerate(candidates):
                status = "✅ চলে" if i in runs else "❌ চলে না"
                match = ""
                if predicted is not None:
                    match = " · টেবিল মেলে" if tables_in(candidate) == predicted else " · টেবিল মেলে না"
                chosen = " ← নির্বাচিত" if candidate == sql else ""
                st.text(f"{i + 1}. [{status}{match}]{chosen}")
                st.code(candidate, language="sql")

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
