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

165 hand-written Bangla question–SQL templates across two difficulty tiers, expanded
in two ways:

- **Value-slot variants (160)** — one literal swapped consistently in the question and
  the SQL: department (গণিত ↔ `'Mathematics'`), CGPA / grade-point / year thresholds
  (৩.৫ ↔ `3.5`), grade, semester, course name, `LIMIT`. Each variant is a new SQL
  target, kept only if it executes and returns rows.
- **Paraphrases** — synonym substitution and register frames; these vary only the
  Bangla wording.

| | train | dev | test |
|---|---|---|---|
| examples | 993 | 213 | 264 |
| templates | 107 | 28 | 30 |
| distinct SQL queries | 224 | 43 | 58 |

**Split design — template-level holdout, stratified by `query_type`.** Every variant
(value or paraphrase) stays with its base template, so nothing derived from a
training template can reach dev or test (0 test SQL queries are seen in training).
Templates are partitioned *within* each `query_type`, so train covers all 28 query
shapes while dev/test consist entirely of unseen templates.

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
| 2 | value-slot augmentation (224 distinct SQL), question-only input, exec-guided | *pending* | | |

Run 1 memorised its 107 SQL targets: train loss reached 0.01 while dev loss rose from
epoch 3, and 49 of its 66 invalid queries were schema-grounding errors
(`no such column`, e.g. `students WHERE grade = 'A+'` with the join dropped). Run 2's
changes target exactly those failure modes. Run 2's test set is larger (264 vs 180)
because value-slot variants of the 30 test templates are included; the test
*templates* are the same 30, so the two runs are comparable at the template level.

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
├── colab_train.ipynb           # Colab notebook — train + evaluate + curves
├── common.py                   # Shared input formatting, config, SQL execution helpers
├── create_database.py          # Schema + synthetic data generator
├── build_dataset.py            # Value-slot + paraphrase augmentation, split pipeline
├── preprocess_check.py         # Tokenizer analysis + normalization + length profiling
├── train.py                    # Seq2Seq fine-tuning
├── evaluate.py                 # Execution accuracy + error analysis
├── app.py                      # Streamlit demo
├── data/
│   ├── templates.json          # 165 hand-crafted Bangla question–SQL pairs
│   ├── dataset_{train,dev,test}.json
│   ├── dataset_stats.json
│   └── train_config.json       # Written by preprocess_check.py
├── logs/                       # Training history, metrics, predictions
├── er_diagram.md               # ER diagram (Mermaid)
└── requirements*.txt
```
