"""
BanglaSQL — Database Schema & Synthetic Data Generator
University Management System: 6 tables

Tables:
    1. departments  — academic departments
    2. instructors   — faculty members (FK → departments)
    3. students      — enrolled students (FK → departments)
    4. courses       — offered courses (FK → departments, instructors)
    5. enrollments   — student-course registrations with grades (FK → students, courses)
    6. attendance    — per-session attendance records (FK → students, courses)

Run this script to create and populate `banglasql.db`.
"""

import sqlite3
import random
import os

from faker import Faker

# Reproducibility
SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banglasql.db")

# ── Bangla-flavored realistic data pools ──────────────────────────────────────

DEPARTMENT_NAMES = [
    "Computer Science and Engineering",
    "Electrical and Electronic Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Business Administration",
    "Economics",
    "English",
    "Mathematics",
    "Physics",
    "Chemistry",
]

BANGLA_FIRST_NAMES = [
    "Rahim", "Karim", "Jamal", "Hasan", "Husain",
    "Ayesha", "Fatima", "Nusrat", "Tahmina", "Sharmin",
    "Arif", "Sohel", "Rafiq", "Nasir", "Tanvir",
    "Mithila", "Priya", "Sumaiya", "Tasnim", "Farhana",
    "Imran", "Sakib", "Mahmudul", "Ashraful", "Rezaul",
    "Nazmul", "Kamrul", "Shahidul", "Mizanur", "Khairul",
    "Rumana", "Laboni", "Tania", "Mou", "Sadia",
    "Nahid", "Rasel", "Joynal", "Babul", "Monir",
]

BANGLA_LAST_NAMES = [
    "Hossain", "Rahman", "Ahmed", "Islam", "Uddin",
    "Alam", "Khan", "Begum", "Akter", "Sultana",
    "Chowdhury", "Miah", "Sarker", "Das", "Barua",
    "Talukder", "Mondal", "Sikder", "Haque", "Kamal",
]

COURSE_PREFIXES = {
    "Computer Science and Engineering": ("CSE", [
        "Introduction to Programming", "Data Structures", "Algorithms",
        "Database Systems", "Operating Systems", "Computer Networks",
        "Software Engineering", "Artificial Intelligence",
        "Machine Learning", "Computer Architecture",
    ]),
    "Electrical and Electronic Engineering": ("EEE", [
        "Circuit Analysis", "Digital Electronics", "Signals and Systems",
        "Electromagnetic Theory", "Power Systems", "Control Systems",
        "Microprocessors", "VLSI Design",
        "Communication Systems", "Renewable Energy",
    ]),
    "Mechanical Engineering": ("ME", [
        "Engineering Mechanics", "Thermodynamics", "Fluid Mechanics",
        "Heat Transfer", "Manufacturing Processes", "Machine Design",
        "Robotics", "Automobile Engineering",
    ]),
    "Civil Engineering": ("CE", [
        "Surveying", "Structural Analysis", "Concrete Technology",
        "Geotechnical Engineering", "Transportation Engineering",
        "Environmental Engineering", "Hydraulics", "Construction Management",
    ]),
    "Business Administration": ("BBA", [
        "Principles of Management", "Marketing Management",
        "Financial Accounting", "Business Statistics",
        "Human Resource Management", "Organizational Behavior",
        "Strategic Management", "Entrepreneurship",
    ]),
    "Economics": ("ECON", [
        "Microeconomics", "Macroeconomics", "Econometrics",
        "Development Economics", "International Economics",
        "Public Finance", "Monetary Economics",
    ]),
    "English": ("ENG", [
        "Introduction to Literature", "Linguistics",
        "Creative Writing", "British Literature",
        "American Literature", "Sociolinguistics",
        "English for Academic Purposes",
    ]),
    "Mathematics": ("MATH", [
        "Calculus I", "Calculus II", "Linear Algebra",
        "Differential Equations", "Probability and Statistics",
        "Discrete Mathematics", "Numerical Methods",
        "Abstract Algebra",
    ]),
    "Physics": ("PHYS", [
        "Classical Mechanics", "Electromagnetism",
        "Quantum Mechanics", "Optics",
        "Statistical Mechanics", "Nuclear Physics",
        "Solid State Physics",
    ]),
    "Chemistry": ("CHEM", [
        "Organic Chemistry", "Inorganic Chemistry",
        "Physical Chemistry", "Analytical Chemistry",
        "Biochemistry", "Polymer Chemistry",
        "Environmental Chemistry",
    ]),
}

SEMESTERS = [
    "Spring 2023", "Summer 2023", "Fall 2023",
    "Spring 2024", "Summer 2024", "Fall 2024",
    "Spring 2025",
]

GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D", "F"]
GRADE_POINTS = {
    "A+": 4.00, "A": 3.75, "A-": 3.50,
    "B+": 3.25, "B": 3.00, "B-": 2.75,
    "C+": 2.50, "C": 2.25, "D": 2.00, "F": 0.00,
}

ATTENDANCE_STATUS = ["Present", "Absent", "Late"]


# ── Schema creation ───────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Departments
CREATE TABLE IF NOT EXISTS departments (
    dept_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    dept_name   TEXT    NOT NULL UNIQUE,
    building    TEXT,
    phone       TEXT
);

-- Instructors
CREATE TABLE IF NOT EXISTS instructors (
    instructor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name      TEXT    NOT NULL,
    last_name       TEXT    NOT NULL,
    email           TEXT    UNIQUE,
    dept_id         INTEGER NOT NULL,
    designation     TEXT    NOT NULL,
    joining_year    INTEGER,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

-- Students
CREATE TABLE IF NOT EXISTS students (
    student_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name  TEXT    NOT NULL,
    last_name   TEXT    NOT NULL,
    email       TEXT    UNIQUE,
    dept_id     INTEGER NOT NULL,
    year_of_admission INTEGER NOT NULL,
    cgpa        REAL    DEFAULT 0.0,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

-- Courses
CREATE TABLE IF NOT EXISTS courses (
    course_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code     TEXT    NOT NULL UNIQUE,
    course_name     TEXT    NOT NULL,
    credits         INTEGER NOT NULL,
    dept_id         INTEGER NOT NULL,
    instructor_id   INTEGER,
    semester        TEXT    NOT NULL,
    FOREIGN KEY (dept_id)       REFERENCES departments(dept_id),
    FOREIGN KEY (instructor_id) REFERENCES instructors(instructor_id)
);

-- Enrollments (grades)
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL,
    course_id       INTEGER NOT NULL,
    grade           TEXT,
    grade_point     REAL,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id)  REFERENCES courses(course_id),
    UNIQUE(student_id, course_id)
);

-- Attendance
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL,
    course_id       INTEGER NOT NULL,
    date            TEXT    NOT NULL,
    status          TEXT    NOT NULL CHECK(status IN ('Present', 'Absent', 'Late')),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id)  REFERENCES courses(course_id)
);
"""


# ── Data population ───────────────────────────────────────────────────────────

def populate(conn: sqlite3.Connection):
    cur = conn.cursor()

    # 1. Departments (10 rows)
    buildings = [
        "Building A", "Building B", "Building C", "Building D", "Building E",
        "Building F", "Building G", "Building H", "Building I", "Building J",
    ]
    for i, dept_name in enumerate(DEPARTMENT_NAMES):
        phone = f"+880-2-{random.randint(1000000, 9999999)}"
        cur.execute(
            "INSERT INTO departments (dept_name, building, phone) VALUES (?, ?, ?)",
            (dept_name, buildings[i], phone),
        )
    conn.commit()

    # 2. Instructors (~50, spread across departments)
    designations = ["Professor", "Associate Professor", "Assistant Professor", "Lecturer"]
    instructor_ids_by_dept = {}  # dept_id → [instructor_ids]
    for dept_id in range(1, 11):
        instructor_ids_by_dept[dept_id] = []
        count = random.randint(4, 6)
        for _ in range(count):
            fn = random.choice(BANGLA_FIRST_NAMES)
            ln = random.choice(BANGLA_LAST_NAMES)
            email = f"{fn.lower()}.{ln.lower()}{random.randint(1,99)}@university.edu.bd"
            designation = random.choice(designations)
            joining = random.randint(2000, 2022)
            cur.execute(
                "INSERT INTO instructors (first_name, last_name, email, dept_id, designation, joining_year) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fn, ln, email, dept_id, designation, joining),
            )
            instructor_ids_by_dept[dept_id].append(cur.lastrowid)
    conn.commit()

    # 3. Students (300 rows, spread across departments)
    student_ids = []
    for dept_id in range(1, 11):
        count = random.randint(25, 35)
        for _ in range(count):
            fn = random.choice(BANGLA_FIRST_NAMES)
            ln = random.choice(BANGLA_LAST_NAMES)
            email = f"{fn.lower()}.{ln.lower()}{random.randint(100,9999)}@student.university.edu.bd"
            year_adm = random.randint(2019, 2024)
            cgpa = round(random.uniform(2.0, 4.0), 2)
            cur.execute(
                "INSERT INTO students (first_name, last_name, email, dept_id, year_of_admission, cgpa) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fn, ln, email, dept_id, year_adm, cgpa),
            )
            student_ids.append(cur.lastrowid)
    conn.commit()

    # 4. Courses (~80, from department course lists)
    course_ids_by_dept = {}
    all_course_ids = []
    for dept_id, dept_name in enumerate(DEPARTMENT_NAMES, start=1):
        prefix, course_names = COURSE_PREFIXES[dept_name]
        course_ids_by_dept[dept_id] = []
        for j, cname in enumerate(course_names):
            code = f"{prefix}{(j + 1) * 100 + random.randint(1, 9)}"
            credits = random.choice([3, 3, 3, 4])  # mostly 3-credit
            instructor_id = random.choice(instructor_ids_by_dept[dept_id])
            semester = random.choice(SEMESTERS)
            cur.execute(
                "INSERT INTO courses (course_code, course_name, credits, dept_id, instructor_id, semester) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (code, cname, credits, dept_id, instructor_id, semester),
            )
            cid = cur.lastrowid
            course_ids_by_dept[dept_id].append(cid)
            all_course_ids.append(cid)
    conn.commit()

    # 5. Enrollments (~500 rows)
    enrollment_pairs = set()
    for _ in range(500):
        sid = random.choice(student_ids)
        # Get student's dept
        cur.execute("SELECT dept_id FROM students WHERE student_id = ?", (sid,))
        s_dept = cur.fetchone()[0]
        # 70% chance to enroll in own department, 30% cross-department
        if random.random() < 0.7 and course_ids_by_dept.get(s_dept):
            cid = random.choice(course_ids_by_dept[s_dept])
        else:
            cid = random.choice(all_course_ids)

        if (sid, cid) in enrollment_pairs:
            continue
        enrollment_pairs.add((sid, cid))

        grade = random.choice(GRADES)
        gp = GRADE_POINTS[grade]
        cur.execute(
            "INSERT INTO enrollments (student_id, course_id, grade, grade_point) "
            "VALUES (?, ?, ?, ?)",
            (sid, cid, grade, gp),
        )
    conn.commit()

    # 6. Attendance (~500 rows)
    for _ in range(500):
        sid = random.choice(student_ids)
        cid = random.choice(all_course_ids)
        # Random date in 2023–2025
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        year = random.choice([2023, 2024, 2025])
        date_str = f"{year}-{month:02d}-{day:02d}"
        status = random.choices(
            ATTENDANCE_STATUS,
            weights=[0.75, 0.15, 0.10],  # mostly present
            k=1,
        )[0]
        cur.execute(
            "INSERT INTO attendance (student_id, course_id, date, status) "
            "VALUES (?, ?, ?, ?)",
            (sid, cid, date_str, status),
        )
    conn.commit()


def print_summary(conn: sqlite3.Connection):
    cur = conn.cursor()
    tables = ["departments", "instructors", "students", "courses", "enrollments", "attendance"]
    print("\n" + "=" * 50)
    print("BanglaSQL Database — Summary")
    print("=" * 50)
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table:<15} : {count:>5} rows")
    print("=" * 50)

    # Sample queries to verify data
    print("\n-- Sample: Top 5 students by CGPA --")
    cur.execute("SELECT first_name, last_name, cgpa FROM students ORDER BY cgpa DESC LIMIT 5")
    for row in cur.fetchall():
        print(f"  {row[0]} {row[1]} — CGPA: {row[2]}")

    print("\n-- Sample: Students per department --")
    cur.execute("""
        SELECT d.dept_name, COUNT(s.student_id) as student_count
        FROM departments d
        LEFT JOIN students s ON d.dept_id = s.dept_id
        GROUP BY d.dept_name
        ORDER BY student_count DESC
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:<45} : {row[1]} students")

    print("\n-- Sample: Average grade point per course (top 5) --")
    cur.execute("""
        SELECT c.course_code, c.course_name, ROUND(AVG(e.grade_point), 2) as avg_gp, COUNT(e.enrollment_id) as enrolled
        FROM courses c
        JOIN enrollments e ON c.course_id = e.course_id
        GROUP BY c.course_id
        HAVING enrolled >= 3
        ORDER BY avg_gp DESC
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  {row[0]} {row[1]:<35} — Avg GP: {row[2]}, Enrolled: {row[3]}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Remove existing DB for a clean rebuild
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database: {DB_PATH}")

    conn = sqlite3.Connection(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    print(f"Creating schema in: {DB_PATH}")
    conn.executescript(SCHEMA_SQL)

    print("Populating with synthetic data...")
    populate(conn)

    print_summary(conn)

    conn.close()
    print(f"\nDone! Database saved to: {DB_PATH}")
