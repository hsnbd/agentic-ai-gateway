"""The mock student information system: schema, seed data, and the rules it enforces.

Everything the MCP tools do goes through ``Registry``, so the rules (seats,
prerequisites, credit cap, who teaches what) are tested without MCP.
"""

from __future__ import annotations

import hashlib
import random
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Most credits a student may carry in one semester.
CREDIT_CAP = 18
#: Below this attendance or average mark a student is "at risk".
AT_RISK_ATTENDANCE = 0.75
AT_RISK_MARK = 50.0
CURRENT_SEMESTER = "Spring 2027"

SCHEMA = """
CREATE TABLE IF NOT EXISTS programs (
    code TEXT PRIMARY KEY, name TEXT NOT NULL, level TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teachers (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL, pin_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS courses (
    code TEXT PRIMARY KEY, title TEXT NOT NULL, program TEXT NOT NULL REFERENCES programs(code),
    credits INTEGER NOT NULL, seats INTEGER NOT NULL, teacher_id TEXT REFERENCES teachers(id),
    prerequisite TEXT REFERENCES courses(code), offered INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS students (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
    program TEXT NOT NULL REFERENCES programs(code), batch INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS enrollments (
    student_id TEXT NOT NULL REFERENCES students(id),
    course_code TEXT NOT NULL REFERENCES courses(code),
    semester TEXT NOT NULL, status TEXT NOT NULL,  -- enrolled, completed, dropped
    grade TEXT, PRIMARY KEY (student_id, course_code, semester)
);
CREATE TABLE IF NOT EXISTS progress (
    student_id TEXT NOT NULL, course_code TEXT NOT NULL,
    attendance REAL NOT NULL, average_mark REAL NOT NULL, assessments_done INTEGER NOT NULL,
    assessments_total INTEGER NOT NULL, PRIMARY KEY (student_id, course_code)
);
"""

PROGRAMS = [
    ("BSSE", "Bachelor of Science in Software Engineering", "undergraduate"),
    ("MSSE", "Master of Science in Software Engineering", "graduate"),
    ("MIT", "Master in Information Technology", "graduate"),
]

#: (id, name, email, pin)
TEACHERS = [
    ("T-01", "Dr. Ayesha Rahman", "ayesha@iit.example.edu", "1111"),
    ("T-02", "Dr. Kamal Hossain", "kamal@iit.example.edu", "2222"),
    ("T-03", "Nusrat Jahan", "nusrat@iit.example.edu", "3333"),
    ("T-04", "Dr. Farhan Ahmed", "farhan@iit.example.edu", "4444"),
]

#: (code, title, program, credits, seats, teacher, prerequisite)
COURSES = [
    ("SE 101", "Introduction to Software Engineering", "BSSE", 3, 60, "T-01", None),
    ("SE 102", "Structured Programming", "BSSE", 3, 60, "T-03", None),
    ("SE 201", "Data Structures", "BSSE", 3, 50, "T-03", "SE 102"),
    ("SE 202", "Object Oriented Design", "BSSE", 3, 50, "T-01", "SE 102"),
    ("SE 301", "Software Requirements Engineering", "BSSE", 3, 45, "T-02", "SE 101"),
    ("SE 302", "Software Testing", "BSSE", 3, 45, "T-03", "SE 201"),
    ("SE 401", "Software Architecture", "BSSE", 3, 40, "T-02", "SE 202"),
    ("SE 402", "Machine Learning for Software Engineering", "BSSE", 3, 2, "T-04", "SE 201"),
    ("SE 501", "Advanced Software Engineering", "MSSE", 3, 30, "T-02", None),
    ("SE 502", "Empirical Software Engineering", "MSSE", 3, 30, "T-04", "SE 501"),
    ("MIT 501", "Information Systems Management", "MIT", 3, 40, "T-04", None),
    ("MIT 502", "Cloud Computing", "MIT", 3, 40, "T-01", None),
    ("MIT 503", "Cyber Security", "MIT", 3, 35, "T-02", "MIT 501"),
]


def _hash(pin: str) -> str:
    return hashlib.sha256(f"iit-mock:{pin}".encode()).hexdigest()


class RuleError(ValueError):
    """A request the institute's rules refuse; the message is shown to the user."""


@dataclass
class Registry:
    path: Path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    # ---- setup ---------------------------------------------------------------

    def seed(self, *, students: int = 40, seed: int = 7) -> None:
        """Recreate the database with deterministic demo data."""
        self.path.unlink(missing_ok=True)
        rng = random.Random(seed)
        with closing(self.connect()) as db, db:
            db.executescript(SCHEMA)
            db.executemany("INSERT INTO programs VALUES (?, ?, ?)", PROGRAMS)
            db.executemany(
                "INSERT INTO teachers VALUES (?, ?, ?, ?)",
                [(t, n, e, _hash(p)) for t, n, e, p in TEACHERS],
            )
            db.executemany(
                "INSERT INTO courses (code, title, program, credits, seats, teacher_id,"
                " prerequisite) VALUES (?, ?, ?, ?, ?, ?, ?)",
                COURSES,
            )
            first = ["Rahim", "Karim", "Fatema", "Sadia", "Tanvir", "Nadia", "Imran", "Mitu"]
            last = ["Uddin", "Akter", "Islam", "Chowdhury", "Sarker", "Haque", "Khan", "Das"]
            for index in range(1, students + 1):
                program = "BSSE" if index <= 28 else ("MSSE" if index <= 34 else "MIT")
                batch = 13 + index % 3 if program == "BSSE" else 4
                name = f"{rng.choice(first)} {rng.choice(last)}"
                student_id = f"S-{index:04d}"
                db.execute(
                    "INSERT INTO students VALUES (?, ?, ?, ?, ?)",
                    (student_id, name, f"s{index:04d}@student.iit.example.edu", program, batch),
                )
                self._seed_history(db, rng, student_id, program, batch)

    def _seed_history(
        self, db: sqlite3.Connection, rng: random.Random, student_id: str, program: str, batch: int
    ) -> None:
        courses = [c for c in COURSES if c[2] == program]
        done = courses[: 2 + (15 - batch)] if program == "BSSE" else courses[:1]
        for code, *_ in done:
            grade = rng.choice(["A+", "A", "A-", "B+", "B", "C"])
            db.execute(
                "INSERT INTO enrollments VALUES (?, ?, 'Fall 2026', 'completed', ?)",
                (student_id, code, grade),
            )
        done_codes = {c[0] for c in done}
        current = [
            c for c in courses if c[0] not in done_codes and (c[6] is None or c[6] in done_codes)
        ]
        for code, *_ in current[:2]:
            db.execute(
                "INSERT INTO enrollments VALUES (?, ?, ?, 'enrolled', NULL)",
                (student_id, code, CURRENT_SEMESTER),
            )
            total = 6
            db.execute(
                "INSERT INTO progress VALUES (?, ?, ?, ?, ?, ?)",
                (
                    student_id,
                    code,
                    round(rng.uniform(0.55, 1.0), 2),
                    round(rng.uniform(35, 95), 1),
                    rng.randint(2, 4),
                    total,
                ),
            )

    # ---- catalogue -----------------------------------------------------------

    def offered_courses(self, program: str | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT c.code, c.title, c.program, c.credits, c.prerequisite, t.name AS teacher,
                   c.seats - (SELECT COUNT(*) FROM enrollments e WHERE e.course_code = c.code
                              AND e.semester = ? AND e.status = 'enrolled') AS seats_left
            FROM courses c LEFT JOIN teachers t ON t.id = c.teacher_id
            WHERE c.offered = 1 AND (? IS NULL OR c.program = ?) ORDER BY c.code
        """
        with closing(self.connect()) as db:
            rows = db.execute(sql, (CURRENT_SEMESTER, program, program)).fetchall()
        return [dict(row) for row in rows]

    # ---- students ------------------------------------------------------------

    def _student(self, db: sqlite3.Connection, student_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        if row is None:
            raise RuleError(f"No student with id {student_id}")
        return row  # type: ignore[no-any-return]

    def profile(self, student_id: str) -> dict[str, Any]:
        with closing(self.connect()) as db:
            student = dict(self._student(db, student_id))
            credits = db.execute(
                "SELECT COALESCE(SUM(c.credits), 0) FROM enrollments e JOIN courses c"
                " ON c.code = e.course_code WHERE e.student_id = ? AND e.status = 'completed'",
                (student_id,),
            ).fetchone()[0]
        return {**student, "completed_credits": credits}

    def enrollments(self, student_id: str) -> list[dict[str, Any]]:
        with closing(self.connect()) as db:
            self._student(db, student_id)
            rows = db.execute(
                "SELECT e.course_code, c.title, c.credits, e.semester, e.status"
                " FROM enrollments e JOIN courses c ON c.code = e.course_code"
                " WHERE e.student_id = ? AND e.status != 'dropped'"
                " ORDER BY e.semester, e.course_code",
                (student_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def grades(self, student_id: str) -> dict[str, Any]:
        with closing(self.connect()) as db:
            self._student(db, student_id)
            completed = db.execute(
                "SELECT course_code, semester, grade FROM enrollments"
                " WHERE student_id = ? AND status = 'completed' ORDER BY course_code",
                (student_id,),
            ).fetchall()
            current = db.execute(
                "SELECT course_code, attendance, average_mark, assessments_done, assessments_total"
                " FROM progress WHERE student_id = ? ORDER BY course_code",
                (student_id,),
            ).fetchall()
        return {
            "completed": [dict(r) for r in completed],
            "in_progress": [dict(r) for r in current],
        }

    def enroll(
        self,
        course_code: str,
        *,
        student_id: str | None = None,
        name: str | None = None,
        email: str | None = None,
        program: str | None = None,
    ) -> dict[str, Any]:
        """Enroll a student; without ``student_id`` a new student is admitted first."""
        code = _course_code(course_code)
        with closing(self.connect()) as db, db:
            course = db.execute("SELECT * FROM courses WHERE code = ?", (code,)).fetchone()
            if course is None or not course["offered"]:
                raise RuleError(f"{code} is not offered this semester")
            created = False
            if student_id is None:
                if not (name and email):
                    raise RuleError("A new student needs a name and an email address")
                student_id = self._admit(db, name, email, program or course["program"])
                created = True
            student = self._student(db, student_id)
            if course["program"] != student["program"]:
                raise RuleError(f"{code} belongs to {course['program']}, not {student['program']}")
            taken = {
                row["course_code"]: row["status"]
                for row in db.execute(
                    "SELECT course_code, status FROM enrollments WHERE student_id = ?",
                    (student_id,),
                )
            }
            if taken.get(code) in ("enrolled", "completed"):
                raise RuleError(f"Already {taken[code]} in {code}")
            prerequisite = course["prerequisite"]
            if prerequisite and taken.get(prerequisite) != "completed":
                raise RuleError(f"{code} requires {prerequisite} to be completed first")
            load = db.execute(
                "SELECT COALESCE(SUM(c.credits), 0) FROM enrollments e JOIN courses c"
                " ON c.code = e.course_code WHERE e.student_id = ? AND e.semester = ?"
                " AND e.status = 'enrolled'",
                (student_id, CURRENT_SEMESTER),
            ).fetchone()[0]
            if load + course["credits"] > CREDIT_CAP:
                raise RuleError(f"Enrolling would exceed the {CREDIT_CAP}-credit semester cap")
            seated = db.execute(
                "SELECT COUNT(*) FROM enrollments WHERE course_code = ? AND semester = ?"
                " AND status = 'enrolled'",
                (code, CURRENT_SEMESTER),
            ).fetchone()[0]
            if seated >= course["seats"]:
                raise RuleError(f"{code} is full")
            db.execute(
                "INSERT OR REPLACE INTO enrollments VALUES (?, ?, ?, 'enrolled', NULL)",
                (student_id, code, CURRENT_SEMESTER),
            )
        return {
            "student_id": student_id,
            "new_student": created,
            "course_code": code,
            "semester": CURRENT_SEMESTER,
            "status": "enrolled",
        }

    def _admit(self, db: sqlite3.Connection, name: str, email: str, program: str) -> str:
        if db.execute("SELECT 1 FROM programs WHERE code = ?", (program,)).fetchone() is None:
            raise RuleError(f"Unknown program {program}")
        if db.execute("SELECT 1 FROM students WHERE email = ?", (email,)).fetchone():
            raise RuleError("A student with that email already exists; sign in as that student")
        count = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        student_id = f"S-{count + 1:04d}"
        batch = 16 if program == "BSSE" else 5
        db.execute(
            "INSERT INTO students VALUES (?, ?, ?, ?, ?)", (student_id, name, email, program, batch)
        )
        return student_id

    def drop(self, course_code: str, student_id: str) -> dict[str, Any]:
        code = _course_code(course_code)
        with closing(self.connect()) as db, db:
            self._student(db, student_id)
            changed = db.execute(
                "UPDATE enrollments SET status = 'dropped' WHERE student_id = ? AND course_code = ?"
                " AND semester = ? AND status = 'enrolled'",
                (student_id, code, CURRENT_SEMESTER),
            ).rowcount
        if not changed:
            raise RuleError(f"Not enrolled in {code} this semester")
        return {"student_id": student_id, "course_code": code, "status": "dropped"}

    # ---- teachers ------------------------------------------------------------

    def verify_teacher(self, teacher_id: str, pin: str) -> dict[str, Any]:
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
        if row is None or row["pin_hash"] != _hash(pin):
            return {"verified": False}
        return {"verified": True, "teacher_id": row["id"], "name": row["name"]}

    def teacher_courses(self, teacher_id: str) -> list[dict[str, Any]]:
        with closing(self.connect()) as db:
            rows = db.execute(
                "SELECT c.code, c.title, c.program, (SELECT COUNT(*) FROM enrollments e WHERE"
                " e.course_code = c.code AND e.semester = ? AND e.status = 'enrolled') AS enrolled"
                " FROM courses c WHERE c.teacher_id = ? ORDER BY c.code",
                (CURRENT_SEMESTER, teacher_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def _taught(self, db: sqlite3.Connection, code: str, teacher_id: str) -> None:
        row = db.execute("SELECT teacher_id FROM courses WHERE code = ?", (code,)).fetchone()
        if row is None:
            raise RuleError(f"No course {code}")
        if row["teacher_id"] != teacher_id:
            raise RuleError(f"You do not teach {code}")

    def course_progress(self, course_code: str, teacher_id: str) -> dict[str, Any]:
        code = _course_code(course_code)
        with closing(self.connect()) as db:
            self._taught(db, code, teacher_id)
            row = db.execute(
                "SELECT COUNT(*) AS students, ROUND(AVG(attendance), 2) AS average_attendance,"
                " ROUND(AVG(average_mark), 1) AS average_mark,"
                " ROUND(AVG(1.0 * assessments_done / assessments_total), 2) AS completion"
                " FROM progress WHERE course_code = ?",
                (code,),
            ).fetchone()
        return {"course_code": code, "semester": CURRENT_SEMESTER, **dict(row)}

    def at_risk(self, course_code: str, teacher_id: str) -> list[dict[str, Any]]:
        code = _course_code(course_code)
        with closing(self.connect()) as db:
            self._taught(db, code, teacher_id)
            rows = db.execute(
                "SELECT s.id AS student_id, s.name, p.attendance, p.average_mark FROM progress p"
                " JOIN students s ON s.id = p.student_id WHERE p.course_code = ?"
                " AND (p.attendance < ? OR p.average_mark < ?) ORDER BY p.average_mark",
                (code, AT_RISK_ATTENDANCE, AT_RISK_MARK),
            ).fetchall()
        return [dict(row) for row in rows]

    def program_progress(self, program: str, teacher_id: str) -> dict[str, Any]:
        """Per-batch summary of a program; open to teachers who teach in it."""
        program = program.strip().upper()
        with closing(self.connect()) as db:
            teaches = db.execute(
                "SELECT 1 FROM courses WHERE program = ? AND teacher_id = ?", (program, teacher_id)
            ).fetchone()
            if teaches is None:
                raise RuleError(f"You do not teach in {program}")
            rows = db.execute(
                "SELECT s.batch, COUNT(DISTINCT s.id) AS students,"
                " ROUND(AVG(p.attendance), 2) AS average_attendance,"
                " ROUND(AVG(p.average_mark), 1) AS average_mark,"
                " SUM(CASE WHEN p.attendance < ? OR p.average_mark < ? THEN 1 ELSE 0 END)"
                " AS at_risk_enrollments"
                " FROM students s LEFT JOIN progress p ON p.student_id = s.id"
                " WHERE s.program = ? GROUP BY s.batch ORDER BY s.batch",
                (AT_RISK_ATTENDANCE, AT_RISK_MARK, program),
            ).fetchall()
        return {
            "program": program,
            "semester": CURRENT_SEMESTER,
            "batches": [dict(r) for r in rows],
        }


def _course_code(raw: str) -> str:
    """Accept "se302", "SE-302" or "SE 302"."""
    compact = "".join(ch for ch in raw.upper() if ch.isalnum())
    letters = "".join(ch for ch in compact if ch.isalpha())
    digits = compact[len(letters) :]
    return f"{letters} {digits}" if letters and digits else raw.strip().upper()
