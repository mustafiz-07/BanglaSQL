# BanglaSQL

Natural Language (Bangla) → SQL query generation for a University Management System.

Fine-tunes **BanglaT5** (`csebuetnlp/banglat5`) on a purpose-built Bangla text-to-SQL
dataset over a 6-table SQLite schema, then evaluates by *executing* the generated
SQL against the database.

## Pipeline

```bash
python create_database.py    # 6-table SQLite schema + synthetic data (Faker)
python build_dataset.py      # templates → paraphrase augmentation → train/dev/test
python preprocess_check.py   # NFC normalization, tokenizer coverage, length profiling
python train.py              # fine-tune BanglaT5 (GPU — use Colab)
python evaluate.py           # execution accuracy + error analysis on the test split
streamlit run app.py         # demo: Bangla question → SQL → result table
```

Training needs a GPU. Open [colab_train.ipynb](colab_train.ipynb) in Google Colab,
set `Runtime → Change runtime type → T4 GPU`, point Step 2 at your repo URL, and
run the cells in order. A 40-epoch run takes roughly 45–70 minutes on a T4.

> Colab installs `requirements_colab.txt`, not `requirements.txt` — Colab already
> ships a working torch/CUDA build and reinstalling a pinned one causes conflicts.

## Dataset

165 hand-written Bangla question–SQL templates across two difficulty tiers, expanded
~6× by rule-based paraphrasing (synonym substitution + register frames) to 990 pairs.

| | train | dev | test |
|---|---|---|---|
| examples | 642 | 168 | 180 |
| templates | 107 | 28 | 30 |

**Split design — template-level holdout, stratified by `query_type`.** Every
augmented variant stays with its base template, so no paraphrase of a training
question can appear in dev or test (0 test SQL queries are seen during training).
Templates are partitioned *within* each `query_type`, so train covers all 28 query
shapes while dev/test consist entirely of unseen templates. That measures
generalization to new questions rather than memorized paraphrases.

Statistics, including leakage and per-split coverage checks, are written to
`data/dataset_stats.json` by `build_dataset.py`.

## Model

Input is a linearized schema appended to the question:

```
translate Bangla to SQL: <বাংলা প্রশ্ন> schema: table: students(student_id, ...) | table: courses(...)
```

`format_input` lives in [common.py](common.py) and is imported by training,
evaluation and the demo, so the prompt cannot drift between them.

Tokenizer coverage on this dataset (measured by `preprocess_check.py`):

| model | vocab | Bangla-script tokens | coverage |
|---|---|---|---|
| `csebuetnlp/banglat5` | 32,100 | 28,644 (89.2%) | **70.5%** |
| `google/mt5-small` | 250,100 | 1,885 (0.8%) | 40.9% |

BanglaT5 needs ~12 tokens per question where mT5 needs ~21, which is why it is the
primary model. To fall back, set `"model_name"` in `data/train_config.json` to
`google/mt5-small` and retrain.

## Evaluation

`evaluate.py` runs each generated query against the database and reports:

- **execution accuracy** (primary) — does the SQL return the gold result set?
  Row order is compared only when the gold query has an `ORDER BY`.
- **exact match** — normalized string equality
- **validity rate** — does the SQL execute at all?
- **component accuracy** — SELECT / WHERE / GROUP BY / HAVING / ORDER BY / JOIN / tables / aggregates
- breakdowns by difficulty and query type, plus a failure-category table
  (`malformed_sql`, `wrong_table`, `wrong_join`, `wrong_aggregate`, `wrong_filter`, …)

```bash
python evaluate.py --split test          # → logs/test_results.json
                                         #   logs/test_predictions.json
```

The database is opened read-only and non-`SELECT` statements are rejected, so a
generated `DROP`/`DELETE` cannot damage it.

## Demo

```bash
pip install streamlit
streamlit run app.py
```

Bangla question in, generated SQL and live result table out. Generated SQL is
validated against the real schema before execution, the connection is read-only,
and empty/invalid results produce a Bangla message rather than a stack trace.

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
├── common.py                   # Shared prompt formatting + config loading
├── create_database.py          # Schema + synthetic data generator
├── build_dataset.py            # Augmentation + split pipeline
├── preprocess_check.py         # Tokenizer analysis + normalization
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
