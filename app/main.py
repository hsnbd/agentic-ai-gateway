"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health
from app.config.settings import Settings, get_settings
from app.core.errors import GatewayError, retry_after_header
from app.core.state import GatewayState

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    state = GatewayState(settings)
    await state.startup()
    app.state.gateway = state
    logger.info(
        "gateway ready: %d deployments, providers=%s",
        len(state.registry.list_deployments()),
        ",".join(state.registry.provider_names),
    )
    try:
        yield
    finally:
        await state.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Agentic AI Gateway",
        description=(
            "Unified middleware between agentic AI applications and LLM providers: "
            "routing, retries and fallbacks, semantic caching, guardrails, RAG, MCP, "
            "and observability."
        ),
        version="0.1.0",
        root_path=settings.root_path,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    from app.observability.logging import configure_logging
    from app.observability.tracing import configure_tracing

    # LOG_LEVEL / LOG_FORMAT and TRACING_ENABLED / OTLP_ENDPOINT only take
    # effect through these; tracing must instrument the app before it serves.
    configure_logging(settings)
    configure_tracing(settings, app)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    _register_routers(app)
    _register_exception_handlers(app)
    _mount_console(app, settings)
    return app


#: (module path, human label, authenticated). Routers are optional so the
#: gateway still boots when a subsystem is disabled or not yet built. The chat
#: dialects authenticate inside the pipeline's auth stage; RAG and MCP expose
#: state-changing routes outside the pipeline, so they are guarded here.
_ROUTER_MODULES: tuple[tuple[str, str, bool], ...] = (
    ("app.api.chat", "openai-dialect", False),
    ("app.api.messages", "anthropic-dialect", False),
    ("app.api.rag", "rag", True),
    ("app.api.mcp", "mcp", True),
    ("app.api.admin", "admin", False),
)


def _register_routers(app: FastAPI) -> None:
    from app.api.deps import require_gateway_principal

    guard = [Depends(require_gateway_principal)]
    #: Labels of the routers actually mounted, for feature reporting.
    app.state.mounted_routers = set()
    for module_path, label, authenticated in _ROUTER_MODULES:
        try:
            module = __import__(module_path, fromlist=["router"])
            app.include_router(module.router, dependencies=guard if authenticated else None)
            app.state.mounted_routers.add(label)
        except (ImportError, AttributeError) as exc:
            logger.info("router %s unavailable (%s); skipping", label, exc)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
        headers: dict[str, str] = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = retry_after_header(exc.retry_after)
        return JSONResponse(
            status_code=exc.status_code, content=exc.to_dict(), headers=headers
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "code": "internal_error",
                }
            },
        )


#: Where `make ui-build` writes the console bundle.
CONSOLE_DIR = os.path.join(os.path.dirname(__file__), "ui_static")


def _mount_console(app: FastAPI, settings: Settings, static_dir: str | None = None) -> None:
    """Serve the built console SPA, when present."""
    if not settings.ui_enabled:
        return

    static_dir = static_dir or CONSOLE_DIR
    index_file = os.path.join(static_dir, "index.html")
    if not os.path.isdir(static_dir) or not os.path.exists(index_file):
        logger.info("console assets not built; %s will 404", settings.ui_path)
        return

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException as StarletteHTTPException

    ui_path = settings.ui_path.rstrip("/")

    class SpaStaticFiles(StaticFiles):
        """Serve index.html for unknown paths so client-side routes work.

        A React router owns paths like `/ui/logs`, which exist only in the
        browser. Without this fallback a refresh or a deep link 404s.

        StaticFiles signals a miss by *raising* ``HTTPException(404)`` rather
        than returning a 404 response, so the fallback has to be in an except
        block. Checking ``response.status_code`` instead silently never fires.
        """

        async def get_response(self, path: str, scope: Any) -> Any:
            try:
                return await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code != 404:
                    raise
                # Missing assets must stay 404 — only unknown *routes* fall
                # back, or a bad bundle URL would return HTML and surface as a
                # confusing MIME-type error in the browser console.
                if "." in path.rsplit("/", 1)[-1]:
                    raise
                return FileResponse(index_file)

    app.mount(ui_path or "/ui", SpaStaticFiles(directory=static_dir, html=True), name="console")


app = create_app()
