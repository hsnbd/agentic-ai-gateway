"""Encryption at rest for credentials the gateway stores on behalf of operators.

Values (not keys) are encrypted with Fernet and stored as ``enc:v1:<token>``,
so a database dump does not expose MCP server tokens and passwords, while the
console can still show which variables and headers are set.

``SECRETS_ENCRYPTION_KEY`` holds one or more comma-separated keys. The first
encrypts; every key decrypts, so a key is rotated by putting the new one first
and keeping the old one until stored values have been rewritten. A key may be
a Fernet key or any high-entropy string (it is hashed into one).

Without a key, values are stored as given. Encrypted values without the key
that wrote them are a configuration error rather than a silent failure.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Mapping

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.core.errors import ErrorCode, GatewayError

PREFIX = "enc:v1:"


def _fernet_key(raw: str) -> bytes:
    try:
        if len(base64.urlsafe_b64decode(raw.encode())) == 32:
            return raw.encode()
    except (binascii.Error, ValueError):
        pass
    return base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest())


class SecretBox:
    def __init__(self, keys: str | None) -> None:
        parsed = [part.strip() for part in (keys or "").split(",") if part.strip()]
        self._fernet = MultiFernet([Fernet(_fernet_key(key)) for key in parsed]) if parsed else None

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    @staticmethod
    def is_encrypted(value: str) -> bool:
        return value.startswith(PREFIX)

    def encrypt(self, value: str) -> str:
        if self._fernet is None or self.is_encrypted(value):
            return value
        return PREFIX + self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: str) -> str:
        if not self.is_encrypted(value):
            return value
        if self._fernet is None:
            raise GatewayError(
                ErrorCode.CONFIGURATION_ERROR,
                "Encrypted secrets are stored but SECRETS_ENCRYPTION_KEY is not set",
            )
        try:
            return self._fernet.decrypt(value[len(PREFIX) :].encode()).decode()
        except InvalidToken:
            raise GatewayError(
                ErrorCode.CONFIGURATION_ERROR,
                "A stored secret cannot be decrypted with SECRETS_ENCRYPTION_KEY",
            ) from None

    def encrypt_map(self, values: Mapping[str, str] | None) -> dict[str, str]:
        return {key: self.encrypt(str(value)) for key, value in (values or {}).items()}

    def decrypt_map(self, values: Mapping[str, str] | None) -> dict[str, str]:
        return {key: self.decrypt(str(value)) for key, value in (values or {}).items()}

    def needs_encryption(self, values: Mapping[str, str] | None) -> bool:
        """True when a key is configured and some stored value is still plaintext."""
        return self.enabled and any(
            not self.is_encrypted(str(value)) for value in (values or {}).values()
        )
