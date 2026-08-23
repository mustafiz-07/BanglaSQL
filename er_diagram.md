# BanglaSQL — ER Diagram

```mermaid
erDiagram
    departments {
        INTEGER dept_id PK
        TEXT dept_name UK
        TEXT building
        TEXT phone
    }

    instructors {
        INTEGER instructor_id PK
        TEXT first_name
        TEXT last_name
        TEXT email UK
        INTEGER dept_id FK
        TEXT designation
        INTEGER joining_year
    }

    students {
        INTEGER student_id PK
        TEXT first_name
        TEXT last_name
        TEXT email UK
        INTEGER dept_id FK
        INTEGER year_of_admission
        REAL cgpa
    }

    courses {
        INTEGER course_id PK
        TEXT course_code UK
        TEXT course_name
        INTEGER credits
        INTEGER dept_id FK
        INTEGER instructor_id FK
        TEXT semester
    }

    enrollments {
        INTEGER enrollment_id PK
        INTEGER student_id FK
        INTEGER course_id FK
        TEXT grade
        REAL grade_point
    }

    attendance {
        INTEGER attendance_id PK
        INTEGER student_id FK
        INTEGER course_id FK
        TEXT date
        TEXT status
    }

    departments ||--o{ instructors : "has"
    departments ||--o{ students : "has"
    departments ||--o{ courses : "offers"
    instructors ||--o{ courses : "teaches"
    students ||--o{ enrollments : "enrolls in"
    courses ||--o{ enrollments : "has"
    students ||--o{ attendance : "has"
    courses ||--o{ attendance : "has"
```

## Table Summary

| Table | Rows | Description |
|---|---|---|
| departments | 10 | Academic departments |
| instructors | 50 | Faculty members |
| students | 304 | Enrolled students |
| courses | 80 | Offered courses |
| enrollments | 471 | Student-course registrations with grades |
| attendance | 500 | Per-session attendance records |

## Relationships

- **departments → instructors**: One department has many instructors
- **departments → students**: One department has many students
- **departments → courses**: One department offers many courses
- **instructors → courses**: One instructor teaches many courses
- **students → enrollments**: One student has many enrollments
- **courses → enrollments**: One course has many enrollments
- **students → attendance**: One student has many attendance records
- **courses → attendance**: One course has many attendance records
