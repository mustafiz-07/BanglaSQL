# BanglaSQL

Natural Language (Bangla) → SQL query generation for a University Management System.

Fine-tunes **BanglaT5** (`csebuetnlp/banglat5`) on a purpose-built Bangla text-to-SQL
dataset over a 6-table SQLite schema, then evaluates by *executing* the generated
SQL against the database.

## Pipeline

```bash
python create_database.py    # 6-table SQLite schema + synthetic data (Faker)
python build_dataset.py      # templates → value-slot variants → paraphrases → train/dev/test
python preprocess_check.py   # NFC normalization, tokenizer coverage, length profiling
python train.py              # fine-tune BanglaT5 (GPU — use Colab)
python evaluate.py           # execution accuracy + error analysis on the test split
streamlit run app.py         # demo: Bangla question → SQL → result table
```

Training needs a GPU. Open [colab_train.ipynb](colab_train.ipynb) in Google Colab,
set `Runtime → Change runtime type → T4 GPU`, point Step 2 at your repo URL, and
run the cells in order. Start from a fresh runtime (`Runtime → Disconnect and delete
runtime`) so Step 2 clones the latest code rather than reusing an old checkout.

> Colab installs `requirements_colab.txt`, not `requirements.txt` — Colab already
> ships a working torch/CUDA build and reinstalling a pinned one causes conflicts.

## Dataset

228 hand-written Bangla question–SQL templates across two difficulty tiers, expanded
in two ways:

- **Value-slot variants (207)** — one literal swapped consistently in the question and
  the SQL: department (গণিত ↔ `'Mathematics'`), CGPA / grade-point / year thresholds
  (৩.৫ ↔ `3.5`), grade, semester, course name, `LIMIT`. Each variant is a new SQL
  target, kept only if it executes and returns rows.
- **Polarity variants (44)** — the comparison operator or sort direction flipped in
  both places: বেশি ↔ কম with `>` ↔ `<`, অবরোহী ↔ আরোহী with `DESC` ↔ `ASC`, and
  "top N" superlatives (সবচেয়ে বেশি ↔ সবচেয়ে কম, সর্বোচ্চ ↔ সর্বনিম্ন) with the
  `ORDER BY` direction. Without these the operator stays welded to its template and
  the model never has to read the Bangla word to get it right; they were run 3's two
  largest error categories. Queries whose direction is carried by `MAX()`/`MIN()`
  are excluded, since those rules do not rewrite the aggregate.

  Coverage after this change: 12 of 16 templates that use a comparison now appear
  with both `>` and `<`, and 21 of 33 that sort appear with both `ASC` and `DESC`.
  Most of the remainder are not flippable — their question states no direction
  ("নাম অনুযায়ী সাজাও"), or the opposite already exists as its own template.
- **Paraphrases** — synonym substitution and register frames; these vary only the
  Bangla wording.

| | train | dev | test |
|---|---|---|---|
| examples | 1521 | 261 | 339 |
| templates | 165 | 29 | 34 |
| distinct SQL queries | 342 | 58 | 79 |

Template counts per `query_type` are deliberately balanced: after run 2, query types
with ≤2 training templates averaged **31.4%** execution accuracy against **54.3%**
for the rest, so 63 templates were added to the thin tail (`limit`, `order_by`,
`order_by_limit`, `select_column`, `select_distinct`, `join`, `multi_join`,
`aggregate_avg/max/min`, …). Every query type that appears in test now has at least
3 training templates.

**Split design — template-level holdout, stratified by `query_type`, stable across
dataset edits.** Every variant (value, polarity or paraphrase) stays with its base
template, so nothing derived from a training template can reach dev or test (0 test
SQL queries are seen in training). Templates are partitioned *within* each
`query_type`, so train covers all 28 query shapes while dev/test consist entirely of
unseen templates.

Assignment is by a template's **position** within its query_type, not by shuffling:
ids are append-only and zero-padded, so a newly written template always sorts last
and every existing template keeps its split. This matters for reading results —
between runs 2 and 3 the old shuffle reassigned templates whenever the template file
changed, which alone moved `join_where_order` from 100% to 0% because test had drawn
a different single template. From run 4 on, per-query-type numbers are comparable
across runs.

Statistics, including leakage and coverage checks, are written to
`data/dataset_stats.json` by `build_dataset.py`.

## Model

The model input is the Bangla question with a task prefix:

```
translate Bangla to SQL: <বাংলা প্রশ্ন>
```

The system has a single fixed database, so a linearized schema would be identical
in every example and carry no conditional information. In run 1 it made up ~220 of
~240 input tokens. Setting `"include_schema": true` in `data/train_config.json`
restores it.

`format_input` and `load_config` live in [common.py](common.py) and are shared by
training, evaluation and the demo. `train.py` saves `banglasql_config.json` next to
the weights, so a checkpoint is always queried with the input format it was trained on.

Tokenizer coverage on this dataset (measured by `preprocess_check.py`):

| model | vocab | Bangla-script tokens | coverage |
|---|---|---|---|
| `csebuetnlp/banglat5` | 32,100 | 28,644 (89.2%) | **67.3%** |
| `google/mt5-small` | 250,100 | 1,885 (0.8%) | 40.0% |

The best checkpoint is selected on dev **execution accuracy** (not exact match, which
rejects equivalent SQL written with a different alias or column order).

## Evaluation

`evaluate.py` decodes with 4-beam search and scores two decoding strategies side by side:

- **top-1 beam** — the model's highest-scoring query
- **execution-guided** — the highest-scoring beam that executes without error
  (Wang et al., 2018); this is the headline number

For each it reports execution accuracy (row order compared only when the gold query
has `ORDER BY`), exact match, validity rate, component accuracy (SELECT / WHERE /
GROUP BY / HAVING / ORDER BY / JOIN / tables / aggregates), breakdowns by difficulty
and query type, and failure categories (`malformed_sql`, `wrong_table`, `wrong_join`,
`wrong_aggregate`, `wrong_filter`, …).

```bash
python evaluate.py --split test          # → logs/test_results.json
                                         #   logs/test_predictions.json
```

The database is opened read-only, non-`SELECT` statements are rejected, and each query
is aborted after 5 seconds.

## Results

| run | setup | test exec. acc. | exact match | validity |
|---|---|---|---|---|
| 1 | 642 train pairs (107 distinct SQL), schema in input, top-1 beam | 27.2% | 21.7% | 63.3% |
| 2 | value-slot augmentation (224 distinct SQL), question-only input, exec-guided | 45.5% | 23.5% | 81.8% |
| 3 | +63 templates for thin query types (301 distinct SQL), LR 2e-4 | 41.8% | 34.0% | 80.8% |
| 4 | polarity augmentation (342 distinct SQL), stable split | 38.0% | 21.8% | 83.2% |
| 5 | duplicate-column rejection, gold-query audit (+ trained 10 epochs longer) | **58.7%** | 34.8% | 91.7% |
| 6 | patience 5→10, orphan ASC/DESC rejection, run-on phrasing repair | 54.6% | 34.8% | 90.9% |
| 7 | dataset audited and rebuilt: 13 gold errors fixed, database repopulated, loopholes closed | 55.0% | **50.4%** | 85.2% |
| 8 | আগে/পরে polarity rule, aggregate aliases removed, target length 128→160 | *pending* | | |

Run 7's component results confirmed the audit: `select` accuracy 44.5% → **75.8%** after
every returned column was named in the question, `having` 40% → **100%** after aliases
were standardised, and exact match 34.8% → **50.4%**, the largest jump in the project.
Execution accuracy stayed flat at 55.0%, within a 95% CI of 0.371–0.727.

**Run 7 breaks comparability on purpose.** An audit of all 228 templates found 13
question/SQL disagreements, 4 questions whose correct answer was an empty table, clauses
that excluded nothing, and answers shared across splits. Gold SQL and the database both
changed, so run 7's numbers stand on their own. See [progress.md](progress.md) for the
full audit and [data/templates_review.json](data/templates_review.json) for the
per-template report.

Runs 4, 5 and 6 share an identical test set. Run 6's targeted fixes each landed —
`order_by` component accuracy 21.3% → 57.3%, `wrong_order_by` failures 15 → 0,
training no longer early-stops — yet the headline fell 4.1 points. The test set holds
339 examples but only **34 distinct templates**, and a cluster bootstrap over templates
gives run 5 a 95% CI of 0.40–0.76 and run 6 one of 0.37–0.71: differences below about
10 points are not measurable on this test set. Read the component table, not the
headline. See [progress.md](progress.md).

Headline numbers are only comparable from run 4 onward — each earlier run rebuilt the
test set (runs 3 and 4 share just 4 of 34 test templates). Run 4's polarity
augmentation cut `wrong_order_by` failures from 39 to 3 and lifted the `order_by`
component from 23.5% to 41.8%, `where` from 54.3% to 65.1% and validity to 83.2%. See
[progress.md](progress.md) for the full run-by-run analysis.

Run 1 memorised its 107 SQL targets: train loss reached 0.01 while dev loss rose from
epoch 3, and 49 of its 66 invalid queries were schema-grounding errors
(`no such column`, e.g. `students WHERE grade = 'A+'` with the join dropped).

Run 2 fixed most of that — execution accuracy 27.2% → 45.5%, validity 63.3% → 81.8%,
and execution-guided decoding added 6.8 points over top-1 beam (38.6% → 45.5%). Its
remaining failures were concentrated in query types with almost no training support:
`limit`, `order_by`, `order_by_limit`, `select_column`, `select_distinct`, `join`,
`multi_join`, `aggregate_avg` and `aggregate_max` all scored 0%, and each had only
1–2 training templates. Run 3 addresses that with template coverage rather than
model changes. Run 2 also showed an instability spike at epoch 10 (dev loss
0.22 → 0.80) at LR 3e-4, hence the drop to 2e-4.

Run 3's headline number (41.8%) is **not** directly comparable to run 2's 45.5%: the
test set was rebuilt with a different template assignment and is weighted toward the
query types run 2 failed outright. The component scores are the fair read, and they
improved substantially — SELECT 40.5% → 68.3%, JOIN 58.7% → 76.4%, aggregates
68.0% → 80.5%, `wrong_table` failures 25 → 6 — as did exact match (23.5% → 34.0%).
The added coverage worked where it was aimed: `limit` and `select_column` went
0% → 100%, `multi_join_where` 13% → 63%, `join` 0% → 50%.

Run 3's remaining errors were concentrated in two categories that augmentation had
never varied: `wrong_filter` (53, 17.8% — `cgpa < 2.5` predicted as `cgpa > 2.5`)
and `wrong_order_by` (39, 13.1%, of which 12 were pure ASC/DESC flips). Run 4 adds
polarity variants for exactly those, and freezes the split so the next comparison is
clean.

## Demo

```bash
pip install streamlit
streamlit run app.py
```

Bangla question in, generated SQL and live result table out. The demo uses the same
execution-guided choice as evaluation, validates table names against the real
schema, uses a read-only connection, and shows Bangla messages for empty or invalid
results.

## Local setup

```bash
pip install -r requirements.txt
# or, with Docker:
docker compose build
docker compose run --rm banglasql bash
```

## Project structure

```
nlp/
├── BanglaSQL_Project_Plan.md   # Full project plan
├── progress.md                 # Run-by-run log: setup, results, failures, fixes
├── colab_train.ipynb           # Colab notebook — train + evaluate + curves
├── common.py                   # Shared input formatting, config, SQL execution helpers
├── create_database.py          # Schema + synthetic data generator
├── build_dataset.py            # Value-slot + paraphrase augmentation, split pipeline
├── preprocess_check.py         # Tokenizer analysis + normalization + length profiling
├── train.py                    # Seq2Seq fine-tuning
├── evaluate.py                 # Execution accuracy + error analysis
├── app.py                      # Streamlit demo
├── data/
│   ├── templates.json          # 228 hand-crafted Bangla question–SQL pairs
│   ├── dataset_{train,dev,test}.json
│   ├── dataset_stats.json
│   └── train_config.json       # Written by preprocess_check.py
├── logs/                       # Training history, metrics, predictions
├── er_diagram.md               # ER diagram (Mermaid)
└── requirements*.txt
```
