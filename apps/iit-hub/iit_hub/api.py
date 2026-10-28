"""HTTP API and chat page.

POST /api/threads                      start (optionally as a teacher)
GET  /api/threads/{id}                 current view
POST /api/threads/{id}/messages        send a message
POST /api/threads/{id}/approve|reject  decide on a pending action

uv run uvicorn iit_hub.api:app --port 8100
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from iit_hub.config import get_settings
from iit_hub.service import (
    ActionPending,
    HubService,
    NothingPending,
    SignInFailed,
    ThreadNotFound,
)

STATIC = Path(__file__).with_name("static")


class StartRequest(BaseModel):
    teacher_id: str | None = None
    pin: str | None = None


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


def create_app(service_factory: Callable[[], HubService] | None = None) -> FastAPI:
    app = FastAPI(title="IIT information hub")
    holder: dict[str, HubService] = {}

    def service() -> HubService:
        if "hub" not in holder:
            holder["hub"] = (
                service_factory or (lambda: HubService.from_settings(get_settings()))
            )()
        return holder["hub"]

    def guarded(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return action()
        except ThreadNotFound:
            raise HTTPException(404, "Unknown thread") from None
        except (ActionPending, NothingPending) as exc:
            raise HTTPException(409, str(exc)) from None
        except SignInFailed as exc:
            raise HTTPException(401, str(exc)) from None

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/api/threads")
    def start(body: StartRequest) -> dict[str, Any]:
        return guarded(lambda: service().start(teacher_id=body.teacher_id, pin=body.pin))

    @app.get("/api/threads/{thread_id}")
    def view(thread_id: str) -> dict[str, Any]:
        return guarded(lambda: service().view(thread_id))

    @app.post("/api/threads/{thread_id}/messages")
    def send(thread_id: str, body: MessageRequest) -> dict[str, Any]:
        return guarded(lambda: service().send(thread_id, body.content))

    @app.post("/api/threads/{thread_id}/approve")
    def approve(thread_id: str) -> dict[str, Any]:
        return guarded(lambda: service().decide(thread_id, True))

    @app.post("/api/threads/{thread_id}/reject")
    def reject(thread_id: str) -> dict[str, Any]:
        return guarded(lambda: service().decide(thread_id, False))

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("iit_hub.api:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
