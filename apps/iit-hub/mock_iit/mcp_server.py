"""IIT's student information system as an MCP server (streamable HTTP).

The gateway registers it as server ``iit``, so its tools reach the hub as
``iit__<tool>``. Identity arguments (``student_id``, ``teacher_id``) are never
chosen by the model: each role's gateway key binds them to the end user (or,
for guests, removes them), see apps/iit-hub/README.md.

    uv run python -m mock_iit.mcp_server --port 4300 --db mock_iit/iit.db
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mock_iit.data import Registry, RuleError


def _checked[T](action: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a registry call; a refused request reaches the model with its reason
    (the SDK hides the text of any other exception)."""
    try:
        return action(*args, **kwargs)
    except RuleError as exc:
        raise ToolError(str(exc)) from exc


def build_server(registry: Registry) -> MCPServer:
    mcp = MCPServer(
        "iit",
        instructions="IIT student information system: courses, enrollment, and progress.",
    )

    @mcp.tool()
    def list_offered_courses(program: str | None = None) -> list[dict[str, Any]]:
        """Courses offered this semester with credits, prerequisite, teacher, and seats left.
        Optionally filter by program code (BSSE, MSSE, MIT)."""
        return _checked(registry.offered_courses, program.strip().upper() if program else None)

    @mcp.tool()
    def enroll_course(
        course_code: str,
        student_id: str | None = None,
        name: str | None = None,
        email: str | None = None,
        program: str | None = None,
    ) -> dict[str, Any]:
        """Enroll the student in a course for the current semester. For someone who is not
        a student yet, give their full name, email, and program: they are admitted and
        enrolled, and the result carries their new student_id."""
        return _checked(
            registry.enroll,
            course_code,
            student_id=student_id,
            name=name,
            email=email,
            program=program,
        )

    @mcp.tool()
    def drop_course(course_code: str, student_id: str) -> dict[str, Any]:
        """Drop one of the student's current-semester courses."""
        return _checked(registry.drop, course_code, student_id)

    @mcp.tool()
    def get_my_profile(student_id: str) -> dict[str, Any]:
        """The student's profile: name, program, batch, completed credits."""
        return _checked(registry.profile, student_id)

    @mcp.tool()
    def get_my_enrollments(student_id: str) -> list[dict[str, Any]]:
        """The student's current and past course enrollments."""
        return _checked(registry.enrollments, student_id)

    @mcp.tool()
    def get_my_grades(student_id: str) -> dict[str, Any]:
        """The student's grades for completed courses and progress in current ones."""
        return _checked(registry.grades, student_id)

    @mcp.tool()
    def verify_teacher(teacher_id: str, pin: str) -> dict[str, Any]:
        """Check a teacher's sign-in PIN."""
        return _checked(registry.verify_teacher, teacher_id, pin)

    @mcp.tool()
    def list_my_courses(teacher_id: str) -> list[dict[str, Any]]:
        """Courses the teacher teaches, with current enrollment counts."""
        return _checked(registry.teacher_courses, teacher_id)

    @mcp.tool()
    def get_course_progress(course_code: str, teacher_id: str) -> dict[str, Any]:
        """Class-level progress in a course the teacher teaches: attendance, average mark,
        and assessment completion."""
        return _checked(registry.course_progress, course_code, teacher_id)

    @mcp.tool()
    def list_at_risk_students(course_code: str, teacher_id: str) -> list[dict[str, Any]]:
        """Students in the teacher's course with attendance under 75% or an average under 50."""
        return _checked(registry.at_risk, course_code, teacher_id)

    @mcp.tool()
    def get_program_progress(program: str, teacher_id: str) -> dict[str, Any]:
        """Per-batch progress of a program (BSSE, MSSE, MIT) the teacher teaches in."""
        return _checked(registry.program_progress, program, teacher_id)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4300)
    parser.add_argument("--db", type=Path, default=Path(__file__).with_name("iit.db"))
    args = parser.parse_args()
    registry = Registry(args.db)
    if not args.db.exists():
        registry.seed()
    build_server(registry).run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
