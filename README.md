# BanglaSQL

Natural Language (Bangla) to SQL Query Generation System for a University Management System domain.

## Setup (Docker)

Build and start the container:

```bash
docker compose build
docker compose run --rm banglasql bash
```

Inside the container, generate the database:

```bash
python create_database.py
```

### Without Docker

```bash
pip install -r requirements.txt
python create_database.py
```

## Database

`banglasql.db` — SQLite database with 6 tables: departments, instructors, students, courses, enrollments, attendance.

## Project Structure

```
nlp/
├── BanglaSQL_Project_Plan.md   # Full project plan
├── create_database.py          # Schema + synthetic data generator
├── banglasql.db                # Generated SQLite database
├── er_diagram.md               # ER diagram (Mermaid)
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Docker Compose config
├── requirements.txt            # Pinned dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```
