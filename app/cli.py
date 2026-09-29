"""Command-line entry point for operating the gateway.

The console UI and the admin API are the normal way to run the gateway, but a
few operations must work *before* either is reachable — creating the schema,
seeding the first administrator, minting a virtual key for a smoke test. Those
live here so an operator is never locked out of a fresh deployment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from decimal import Decimal
from typing import Any

from app.config.settings import get_settings
from app.db.session import Database


async def _with_database(fn: Any) -> Any:
    """Run a coroutine against a started database, always shutting it down."""
    db = Database(get_settings())
    await db.startup()
    try:
        return await fn(db)
    finally:
        await db.shutdown()


async def _cmd_init_db(args: argparse.Namespace) -> int:
    async def run(db: Database) -> int:
        await db.create_all()
        print("Schema created.")
        return 0

    return int(await _with_database(run))


async def _cmd_create_admin(args: argparse.Namespace) -> int:
    from app.auth.console import ConsoleAuthService

    async def run(db: Database) -> int:
        service = ConsoleAuthService(db)
        user = await service.create_user(
            email=args.email, password=args.password, role=args.role
        )
        print(f"Created {user.role} {user.email} (id {user.id}).")
        return 0

    return int(await _with_database(run))


async def _cmd_create_key(args: argparse.Namespace) -> int:
    from app.auth.keys import KeyService

    async def run(db: Database) -> int:
        service = KeyService(db, redis=None)
        _, raw_key = await service.create_key(
            name=args.name,
            max_budget_usd=Decimal(str(args.budget)) if args.budget is not None else None,
            rpm_limit=args.rpm,
            allowed_models=args.model or None,
        )
        # The plaintext key exists only here; it is stored hashed.
        print(raw_key)
        return 0

    return int(await _with_database(run))


def _walk_routes(routes: Any) -> list[dict[str, Any]]:
    """Flatten Starlette/FastAPI routes, descending into included routers.

    FastAPI wraps `include_router` results in a nested router object rather
    than splicing the routes into the parent list, so a flat iteration only
    ever sees the app's own routes.
    """
    rows: list[dict[str, Any]] = []
    for route in routes:
        nested = getattr(route, "original_router", None)
        if nested is not None:
            rows.extend(_walk_routes(nested.routes))
            continue
        path = getattr(route, "path", None)
        if path is None:
            continue
        rows.append({"path": str(path), "methods": sorted(getattr(route, "methods", []) or [])})
    return rows


def _cmd_routes(args: argparse.Namespace) -> int:
    from app.main import create_app

    app = create_app()
    rows = _walk_routes(app.routes)
    rows.sort(key=lambda row: str(row["path"]))
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            methods = ",".join(str(m) for m in row["methods"]) or "-"
            print(f"{methods:<24} {row['path']}")
    print(f"\n{len(rows)} routes", file=sys.stderr)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from app.config.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        factory=False,
    )
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    from app.db import migrate

    migrate.upgrade(args.revision)
    print(f"Database migrated to {args.revision}.")
    return 0


def _cmd_db_stamp(args: argparse.Namespace) -> int:
    from app.db import migrate

    migrate.stamp(args.revision)
    print(f"Database stamped at {args.revision} (no migrations were run).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aigateway", description="Agentic AI Gateway")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the gateway with uvicorn")
    serve.add_argument("--host", default=None, help="Defaults to HOST (0.0.0.0)")
    serve.add_argument("--port", type=int, default=None, help="Defaults to PORT (4000)")
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve, is_async=False)

    init_db = sub.add_parser(
        "init-db", help="Create missing tables directly (development; prefer `migrate`)"
    )
    init_db.set_defaults(func=_cmd_init_db, is_async=True)

    migrate = sub.add_parser("migrate", help="Apply database migrations (Alembic)")
    migrate.add_argument("--revision", default="head")
    migrate.set_defaults(func=_cmd_migrate, is_async=False)

    db_stamp = sub.add_parser(
        "db-stamp",
        help="Mark an existing schema (e.g. made by AUTO_CREATE_SCHEMA) as migrated",
    )
    db_stamp.add_argument("--revision", default="head")
    db_stamp.set_defaults(func=_cmd_db_stamp, is_async=False)

    create_admin = sub.add_parser("create-admin", help="Create a console account")
    create_admin.add_argument("--email", required=True)
    create_admin.add_argument("--password", required=True)
    create_admin.add_argument("--role", default="admin", choices=["admin", "viewer"])
    create_admin.set_defaults(func=_cmd_create_admin, is_async=True)

    create_key = sub.add_parser("create-key", help="Mint a virtual API key")
    create_key.add_argument("--name", required=True)
    create_key.add_argument("--budget", type=float, default=None, help="Max spend in USD")
    create_key.add_argument("--rpm", type=int, default=None, help="Requests per minute")
    create_key.add_argument(
        "--model", action="append", help="Allowed model (repeatable); omit for all"
    )
    create_key.set_defaults(func=_cmd_create_key, is_async=True)

    routes = sub.add_parser("routes", help="List the mounted HTTP routes")
    routes.add_argument("--json", action="store_true")
    routes.set_defaults(func=_cmd_routes, is_async=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.is_async:
        return int(asyncio.run(args.func(args)))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
