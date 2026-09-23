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

# Summer terms are small, so course counts per semester range from 2 to ~20.
SEMESTER_WEIGHTS = [5, 1, 5, 6, 1, 4, 3]

GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D", "F"]
GRADE_POINTS = {
    "A+": 4.00, "A": 3.75, "A-": 3.50,
    "B+": 3.25, "B": 3.00, "B-": 2.75,
    "C+": 2.50, "C": 2.25, "D": 2.00, "F": 0.00,
}

ATTENDANCE_STATUS = ["Present", "Absent", "Late"]

# Every department gets its own CGPA centre, so department averages actually differ.
# Without this, uniform(2.0, 4.0) gives every department a mean near 3.0 and questions
# like "which departments average above 3.5?" have no answer at all.
DEPT_CGPA_MEAN = {
    1: 3.70, 2: 3.64, 3: 3.05, 4: 2.88, 5: 3.28,
    6: 3.10, 7: 3.66, 8: 3.36, 9: 3.15, 10: 2.98,
}

# Students do fail courses, and grades that only ever scatter around a student's CGPA
# never reach F — which left "who got an F?" with no answer at all.
FAILURE_RATE = 0.05

# Fixed dates every course meets on, so date filters land on real sessions instead of
# depending on a random draw. The first two are referenced by name in the templates.
ANCHOR_DATES = ["2024-03-15", "2023-01-10"]
SESSION_DATES = ANCHOR_DATES + [
    "2023-02-14", "2023-04-11", "2023-09-05", "2023-11-21",
    "2024-01-18", "2024-05-09", "2024-08-22", "2024-10-17",
    "2025-02-06", "2025-04-24",
]

# A fifth of students attend poorly. A single flat absence rate leaves nobody with more
# than one or two absences, which makes "who was absent more than 5 times?" unanswerable.
FREQUENT_ABSENTEE_SHARE = 0.20


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
    # Every department is given at least one Professor and one Lecturer, so questions
    # like "who are the professors in Mathematics?" always have an answer.
    designations = ["Professor", "Associate Professor", "Assistant Professor", "Lecturer"]
    pending = []
    for dept_id in range(1, 11):
        count = random.randint(5, 7)
        titles = ["Professor", "Lecturer"] + [random.choice(designations) for _ in range(count - 2)]
        for designation in titles:
            fn = random.choice(BANGLA_FIRST_NAMES)
            ln = random.choice(BANGLA_LAST_NAMES)
            email = f"{fn.lower()}.{ln.lower()}{random.randint(1,999)}@university.edu.bd"
            pending.append((fn, ln, email, dept_id, designation, random.randint(2000, 2022)))
    # Shuffle before inserting. Inserting department by department made instructor_id 1..6
    # the whole of department 1, so "any 5 instructors" and "the instructors in Computer
    # Science" returned the same rows and either query answered both questions.
    random.shuffle(pending)
    instructor_ids_by_dept = {d: [] for d in range(1, 11)}
    for row in pending:
        cur.execute(
            "INSERT INTO instructors (first_name, last_name, email, dept_id, designation, joining_year) "
            "VALUES (?, ?, ?, ?, ?, ?)", row)
        instructor_ids_by_dept[row[3]].append(cur.lastrowid)
    conn.commit()

    # 3. Students (~300 rows, spread across departments)
    student_ids = []
    dept_of_student = {}
    cgpa_of_student = {}
    pending = []
    for dept_id in range(1, 11):
        for _ in range(random.randint(25, 35)):
            fn = random.choice(BANGLA_FIRST_NAMES)
            ln = random.choice(BANGLA_LAST_NAMES)
            email = f"{fn.lower()}.{ln.lower()}{random.randint(100,99999)}@student.university.edu.bd"
            # Resample rather than clip: clipping piles a dozen students onto exactly
            # 4.00 and makes "who has exactly 4.0?" a different question than intended.
            cgpa = random.gauss(DEPT_CGPA_MEAN[dept_id], 0.35)
            while not 2.0 <= cgpa <= 4.0:
                cgpa = random.gauss(DEPT_CGPA_MEAN[dept_id], 0.35)
            pending.append((fn, ln, email, dept_id, random.randint(2019, 2024), round(cgpa, 2)))
    # Shuffled for the same reason as instructors: otherwise "the first 10 students" is
    # exactly "the students of department 1".
    random.shuffle(pending)
    for row in pending:
        cur.execute(
            "INSERT INTO students (first_name, last_name, email, dept_id, year_of_admission, cgpa) "
            "VALUES (?, ?, ?, ?, ?, ?)", row)
        student_ids.append(cur.lastrowid)
        dept_of_student[cur.lastrowid] = row[3]
        cgpa_of_student[cur.lastrowid] = row[5]
    # No student is pinned to exactly 4.00. A forced perfect score made the highest CGPA
    # and the highest grade point both 4.0, so "what is the highest grade point?" (test)
    # and "what is the highest CGPA?" (train) had the same answer and the test question
    # could be answered by reciting the training one.
    #
    # The top CGPA is placed outside the strongest department for the same reason: if the
    # best student overall were also a Computer Science student, "the highest CGPA in
    # Computer Science" and "the highest CGPA" would be the same number and a query that
    # dropped the department filter would score correct.
    top = max((sid for sid in student_ids if dept_of_student[sid] == 7),
              key=lambda sid: cgpa_of_student[sid])
    cur.execute("UPDATE students SET cgpa = 3.99 WHERE student_id = ?", (top,))
    cgpa_of_student[top] = 3.99
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
            # Skewed on purpose: with courses spread evenly, every semester clears any
            # threshold a question can ask about and the filter stops discriminating.
            semester = random.choices(SEMESTERS, weights=SEMESTER_WEIGHTS, k=1)[0]
            cur.execute(
                "INSERT INTO courses (course_code, course_name, credits, dept_id, instructor_id, semester) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (code, cname, credits, dept_id, instructor_id, semester),
            )
            cid = cur.lastrowid
            course_ids_by_dept[dept_id].append(cid)
            all_course_ids.append(cid)
    conn.commit()

    # 5. Enrollments — every student takes 2–6 courses, mostly in their own department.
    # Driving this from students rather than sampling at random guarantees that no student
    # and no course ends up with an empty record, which is what made "student 2's
    # attendance" and "courses with more than 5 students" return nothing.
    enrollments_by_student = {}
    enrolled_pairs = set()

    def enrol(sid, cid):
        if (sid, cid) in enrolled_pairs:
            return
        enrolled_pairs.add((sid, cid))
        # A student's course grades scatter around their CGPA instead of being drawn at
        # random. Random grades left a 3.9 student as likely to fail as a 2.1 student,
        # and left nobody scoring A+ twice, so DISTINCT never removed a row.
        if random.random() < FAILURE_RATE:
            grade = "F"
        else:
            target = random.gauss(cgpa_of_student[sid], 0.45)
            grade = min(GRADES, key=lambda g: abs(GRADE_POINTS[g] - target))
        cur.execute(
            "INSERT INTO enrollments (student_id, course_id, grade, grade_point) VALUES (?, ?, ?, ?)",
            (sid, cid, grade, GRADE_POINTS[grade]),
        )
        enrollments_by_student.setdefault(sid, []).append(cid)

    # Courses differ in popularity. With uniform choice every course lands within a few
    # students of every other, so "courses with more than 5 students" selects all 80 and
    # a prediction that drops the HAVING clause scores exactly as well as the gold.
    popularity = {cid: random.random() ** 2.5 + 0.03 for cid in all_course_ids}
    for sid in student_ids:
        own = course_ids_by_dept[dept_of_student[sid]]
        for _ in range(random.randint(2, 6)):
            pool = own if random.random() < 0.75 else all_course_ids
            cid = random.choices(pool, weights=[popularity[c] for c in pool], k=1)[0]
            enrol(sid, cid)
    # Top every course up to at least 3 enrolments so no course is empty. Deliberately
    # not higher: "courses with more than 5 students" has to exclude something, or the
    # HAVING clause is decoration and a query that omits it scores just as well.
    for cid in all_course_ids:
        taken = {s for (s, c) in enrolled_pairs if c == cid}
        while len(taken) < 3:
            sid = random.choice(student_ids)
            if sid not in taken:
                enrol(sid, cid)
                taken.add(sid)
    conn.commit()

    # 6. Attendance — real sessions of courses the student is actually enrolled in.
    # Previously a student could have attendance for a course they never took.
    absentees = set(random.sample(student_ids, int(len(student_ids) * FREQUENT_ABSENTEE_SHARE)))
    for sid, courses in enrollments_by_student.items():
        # A frequent absentee misses roughly 45% of sessions; everyone else about 10%.
        weights = [0.45, 0.40, 0.15] if sid in absentees else [0.82, 0.10, 0.08]
        for cid in courses:
            for date_str in random.sample(SESSION_DATES, random.randint(3, 5)):
                status = random.choices(ATTENDANCE_STATUS, weights=weights, k=1)[0]
                cur.execute(
                    "INSERT INTO attendance (student_id, course_id, date, status) VALUES (?, ?, ?, ?)",
                    (sid, cid, date_str, status),
                )
    # Guarantee the two dates the templates name by hand are populated on both sides.
    for date_str in ANCHOR_DATES:
        for sid in student_ids[:40]:
            cid = enrollments_by_student[sid][0]
            cur.execute(
                "SELECT 1 FROM attendance WHERE student_id = ? AND course_id = ? AND date = ?",
                (sid, cid, date_str),
            )
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO attendance (student_id, course_id, date, status) VALUES (?, ?, ?, 'Present')",
                    (sid, cid, date_str),
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
