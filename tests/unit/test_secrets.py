"""SecretBox: Fernet encryption of stored credential values, with key rotation."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from app.core.errors import GatewayError
from app.core.secrets import PREFIX, SecretBox


def test_disabled_box_passes_values_through() -> None:
    box = SecretBox(None)
    assert not box.enabled
    assert box.encrypt("plain") == "plain"
    assert box.decrypt("plain") == "plain"
    assert not box.needs_encryption({"a": "plain"})
    with pytest.raises(GatewayError, match="not set"):
        box.decrypt(PREFIX + "anything")


def test_round_trip_with_passphrase_and_fernet_keys() -> None:
    for key in ("a long passphrase", Fernet.generate_key().decode()):
        box = SecretBox(key)
        token = box.encrypt("s3cret")
        assert token.startswith(PREFIX) and "s3cret" not in token
        assert box.encrypt(token) == token  # already encrypted: unchanged
        assert box.decrypt(token) == "s3cret"
        assert box.decrypt("legacy plaintext") == "legacy plaintext"


def test_maps_and_needs_encryption() -> None:
    box = SecretBox("k")
    encrypted = box.encrypt_map({"TOKEN": "x", "PORT": 8080})  # type: ignore[dict-item]
    assert box.decrypt_map(encrypted) == {"TOKEN": "x", "PORT": "8080"}
    assert box.encrypt_map(None) == {} and box.decrypt_map(None) == {}
    assert box.needs_encryption({"a": "plain"})
    assert not box.needs_encryption(encrypted)
    assert not box.needs_encryption(None)


def test_rotation_decrypts_with_old_keys_and_rejects_unknown_ones() -> None:
    old = SecretBox("old-key")
    token = old.encrypt("value")
    rotated = SecretBox("new-key, old-key")
    assert rotated.decrypt(token) == "value"
    assert SecretBox("new-key").decrypt(rotated.encrypt("v2")) == "v2"
    with pytest.raises(GatewayError, match="cannot be decrypted"):
        SecretBox("new-key").decrypt(token)
    assert SecretBox(" , ").enabled is False


def test_non_base64_key_is_hashed() -> None:
    assert SecretBox("not base64 !!").decrypt(SecretBox("not base64 !!").encrypt("x")) == "x"
