"""Hub settings, from ``HUB_*`` environment variables or ``apps/iit-hub/.env``.

``mock_iit.seed`` provisions the gateway and writes the role keys to ``.env``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUB_", env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    gateway_url: str = "http://localhost:4000"
    #: Chat model the gateway serves (any name in its config/models.yaml, or "auto").
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    guest_key: SecretStr = SecretStr("")
    student_key: SecretStr = SecretStr("")
    teacher_key: SecretStr = SecretStr("")
    #: LangGraph checkpoints: conversation history and who each user is.
    threads_db: Path = ROOT / "threads.db"
    #: Chunks retrieved per question.
    top_k: int = 4
    host: str = "127.0.0.1"
    port: int = 8100

    def key_for(self, role: str) -> str:
        return {
            "guest": self.guest_key,
            "student": self.student_key,
            "teacher": self.teacher_key,
        }[role].get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
