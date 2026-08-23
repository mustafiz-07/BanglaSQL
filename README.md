# BanglaSQL

Natural Language (Bangla) to SQL Query Generation System for a University Management System domain.

## Training on Google Colab (recommended)

1. Open [colab_train.ipynb](colab_train.ipynb) in Google Colab
2. Set runtime: `Runtime → Change runtime type → T4 GPU`
3. In **Step 2**, replace the repo URL with your GitHub URL
4. Run all cells in order

> **Important:** The notebook uses `requirements_colab.txt` (not `requirements.txt`) to avoid version conflicts with Colab's pre-installed packages (torch, numpy, etc.).

## Local Setup (Docker)

Build and start the container:

```bash
docker compose build
docker compose run --rm banglasql bash
```

Inside the container:

```bash
python create_database.py    # create database
python build_dataset.py      # build dataset splits
python preprocess_check.py   # tokenizer analysis
# Training requires GPU — use Colab for this step
```

### Without Docker

```bash
pip install -r requirements.txt
python create_database.py
python build_dataset.py
python preprocess_check.py
```

## Requirements Files

| File | Use for |
|---|---|
| `requirements.txt` | Local dev / Docker (strict pinned versions) |
| `requirements_colab.txt` | Google Colab (flexible bounds, avoids conflicts) |

## Project Structure

```
nlp/
├── BanglaSQL_Project_Plan.md   # Full project plan
├── colab_train.ipynb           # Colab training notebook (start here)
├── create_database.py          # Schema + synthetic data generator
├── build_dataset.py            # Dataset augmentation + split pipeline
├── preprocess_check.py         # Tokenizer analysis + normalization
├── train.py                    # Seq2Seq training script
├── data/
│   └── templates.json          # 120 hand-crafted Bangla question-SQL pairs
├── er_diagram.md               # ER diagram (Mermaid)
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Docker Compose config
├── requirements.txt            # Pinned deps (local/Docker)
├── requirements_colab.txt      # Flexible deps (Colab)
└── README.md                   # This file
```
