"""Seed the mock IIT system and provision the gateway for the hub.

Idempotent; every run converges on the same setup:

1. the SQLite student information system (``--reset`` recreates it);
2. a gateway team ``iit-hub`` and an ops key for it;
3. the MCP server ``iit`` (the mock system, which must be running);
4. RAG collections ``iit-public``, ``iit-student``, ``iit-teacher`` owned by the
   team, filled from ``mock_iit/docs/{public,student,teacher}``;
5. one gateway key per role (``ROLE_KEYS``), recreated so ``.env`` holds
   their secrets.

    uv run python -m mock_iit.seed --gateway http://localhost:4000 --mcp-url http://127.0.0.1:4300/mcp
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import httpx

from mock_iit.data import Registry

HERE = Path(__file__).resolve().parent
TEAM = "iit-hub"
SERVER = "iit"
COLLECTIONS = {"iit-public": "public", "iit-student": "student", "iit-teacher": "teacher"}

#: Each role's gateway key: what it may read and call, and which tool
#: arguments the gateway fills from the end user instead of the model.
ROLE_KEYS: dict[str, dict[str, Any]] = {
    "guest": {
        "allowed_collections": ["iit-public"],
        "allowed_tools": ["iit__list_offered_courses", "iit__enroll_course"],
        # A guest has no records: enrolling always admits a new student.
        "tool_arg_bindings": {"iit__*": {"student_id": "none", "teacher_id": "none"}},
        "rpm_limit": 30,
    },
    "student": {
        "allowed_collections": ["iit-public", "iit-student"],
        "allowed_tools": [
            "iit__list_offered_courses",
            "iit__get_my_*",
            "iit__enroll_course",
            "iit__drop_course",
        ],
        "tool_arg_bindings": {"iit__*": {"student_id": "end_user", "teacher_id": "none"}},
        "rpm_limit": 60,
    },
    "teacher": {
        "allowed_collections": ["iit-public", "iit-teacher"],
        "allowed_tools": [
            "iit__verify_teacher",
            "iit__list_my_courses",
            "iit__get_course_progress",
            "iit__get_program_progress",
            "iit__list_at_risk_students",
        ],
        "tool_arg_bindings": {"iit__*": {"teacher_id": "end_user", "student_id": "none"}},
        "rpm_limit": 60,
    },
}


class Gateway:
    def __init__(self, base_url: str, master_key: str, email: str, password: str) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=120)
        self.master = {"Authorization": f"Bearer {master_key}"}
        login = self._check(
            self.http.post("/admin/api/auth/login", json={"email": email, "password": password})
        )
        self.admin = {"Authorization": f"Bearer {login['access_token']}"}

    @staticmethod
    def _check(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            raise SystemExit(
                f"{response.request.method} {response.request.url} -> "
                f"{response.status_code}: {response.text}"
            )
        return response.json() if response.content else None

    def call(self, method: str, path: str, headers: dict[str, str], **kwargs: Any) -> Any:
        return self._check(self.http.request(method, path, headers=headers, **kwargs))

    def team(self) -> str:
        teams = self.call("GET", "/admin/api/teams", self.admin, params={"limit": 200})["items"]
        existing = next((t for t in teams if t["name"] == TEAM), None)
        if existing:
            return str(existing["id"])
        created = self.call(
            "POST",
            "/admin/api/teams",
            self.admin,
            json={"name": TEAM, "description": "IIT information hub", "max_budget_usd": 20},
        )
        return str(created["id"])

    def replace_key(self, name: str, **fields: Any) -> str:
        keys = self.call("GET", "/admin/api/keys", self.admin, params={"limit": 200})["items"]
        for key in keys:
            if key["name"] == name:
                self.call("DELETE", f"/admin/api/keys/{key['id']}", self.admin)
        created = self.call("POST", "/admin/api/keys", self.admin, json={"name": name, **fields})
        return str(created["key"])

    def mcp_server(self, url: str) -> None:
        servers = self.call("GET", "/v1/mcp/servers", self.master)
        existing = next((s for s in servers if s["name"] == SERVER), None)
        body = {"name": SERVER, "transport": "http", "url": url, "description": "IIT SIS (mock)"}
        if existing:
            self.call("PATCH", f"/v1/mcp/servers/{existing['id']}", self.master, json=body)
        else:
            self.call("POST", "/v1/mcp/servers", self.master, json=body)

    def collections(self, ops: dict[str, str], embedding_model: str | None) -> None:
        listed = self.call("GET", "/v1/rag/collections", ops)
        for collection in listed:
            if collection["name"] in COLLECTIONS:
                self.call("DELETE", f"/v1/rag/collections/{collection['id']}", ops)
        for name, folder in COLLECTIONS.items():
            body: dict[str, Any] = {"name": name, "description": f"IIT {folder} documents"}
            if embedding_model:
                body["embedding_model"] = embedding_model
            created = self.call("POST", "/v1/rag/collections", ops, json=body)
            for document in sorted((HERE / "docs" / folder).glob("*.md")):
                self.call(
                    "POST",
                    f"/v1/rag/collections/{created['id']}/documents",
                    ops,
                    json={
                        "content": document.read_text(),
                        "source": document.name,
                        "content_type": "text/markdown",
                    },
                )
            print(f"  collection {name}: {len(list((HERE / 'docs' / folder).glob('*.md')))} docs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--gateway", default=os.environ.get("HUB_GATEWAY_URL", "http://localhost:4000")
    )
    parser.add_argument(
        "--master-key", default=os.environ.get("MASTER_KEY", "sk-gateway-master-change-me")
    )
    parser.add_argument(
        "--admin-email", default=os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "admin@local")
    )
    parser.add_argument(
        "--admin-password", default=os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "admin")
    )
    parser.add_argument("--mcp-url", default="http://127.0.0.1:4300/mcp")
    parser.add_argument("--model", default=os.environ.get("HUB_MODEL", "gpt-4o-mini"))
    parser.add_argument("--embedding-model", help="Embedding model for the collections")
    parser.add_argument("--db", type=Path, default=HERE / "iit.db")
    parser.add_argument("--reset", action="store_true", help="Recreate the student database")
    parser.add_argument("--env-file", type=Path, default=HERE.parent / ".env")
    args = parser.parse_args()

    registry = Registry(args.db)
    if args.reset or not args.db.exists():
        registry.seed()
        print(f"Seeded {args.db}")

    gateway = Gateway(args.gateway, args.master_key, args.admin_email, args.admin_password)
    team_id = gateway.team()
    ops = gateway.replace_key("iit-hub-ops", team_id=team_id)
    gateway.mcp_server(args.mcp_url)
    print(f"Registered MCP server {SERVER} -> {args.mcp_url}")
    gateway.collections({"Authorization": f"Bearer {ops}"}, args.embedding_model)

    secrets: dict[str, str] = {}
    for role, scopes in ROLE_KEYS.items():
        secrets[role] = gateway.replace_key(
            f"iit-hub-{role}",
            team_id=team_id,
            allowed_mcp_servers=[SERVER],
            # Contact details are collected on purpose; see config/guardrails.yaml.
            guardrail_policy="user-data",
            max_budget_usd=5,
            **scopes,
        )
        print(f"  key iit-hub-{role}: {scopes['allowed_collections']}")

    args.env_file.write_text(
        f"HUB_GATEWAY_URL={args.gateway}\nHUB_MODEL={args.model}\n"
        + "".join(f"HUB_{role.upper()}_KEY={secret}\n" for role, secret in secrets.items())
    )
    print(f"Wrote {args.env_file}")


if __name__ == "__main__":
    main()
