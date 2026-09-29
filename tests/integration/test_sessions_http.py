"""Console sessions: refresh-token rotation and server-side sign-out."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.integration.conftest import ADMIN_EMAIL, ADMIN_PASSWORD


def _login(client: TestClient, email: str = ADMIN_EMAIL, password: str = ADMIN_PASSWORD) -> Any:
    response = client.post("/admin/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return response.json()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_login_returns_a_refresh_token(client: TestClient) -> None:
    session = _login(client)
    assert session["refresh_token"]
    assert session["refresh_expires_in"] > session["expires_in"]


def test_refresh_rotates_the_pair(client: TestClient) -> None:
    session = _login(client)
    refreshed = client.post(
        "/admin/api/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )
    assert refreshed.status_code == 200, refreshed.text
    new = refreshed.json()
    assert new["access_token"] != session["access_token"]
    assert client.get("/admin/api/auth/me", headers=_bearer(new["access_token"])).is_success

    # A refresh token works exactly once.
    replay = client.post(
        "/admin/api/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )
    assert replay.status_code == 401


def test_access_tokens_cannot_be_used_to_refresh(client: TestClient) -> None:
    session = _login(client)
    response = client.post(
        "/admin/api/auth/refresh", json={"refresh_token": session["access_token"]}
    )
    assert response.status_code == 401


def test_logout_revokes_the_session(client: TestClient) -> None:
    session = _login(client)
    headers = _bearer(session["access_token"])
    logout = client.post(
        "/admin/api/auth/logout",
        json={"refresh_token": session["refresh_token"]},
        headers=headers,
    )
    assert logout.status_code == 200
    assert client.get("/admin/api/auth/me", headers=headers).status_code == 401
    # The data-plane routes honour console revocation too.
    assert client.get("/v1/rag/collections", headers=headers).status_code == 401
    refresh = client.post(
        "/admin/api/auth/refresh", json={"refresh_token": session["refresh_token"]}
    )
    assert refresh.status_code == 401


def test_logout_only_ends_that_session(client: TestClient) -> None:
    first, second = _login(client), _login(client)
    client.post("/admin/api/auth/logout", headers=_bearer(first["access_token"]))
    assert client.get("/admin/api/auth/me", headers=_bearer(second["access_token"])).is_success


def test_password_change_signs_out_everywhere(client: TestClient) -> None:
    other = _login(client)
    session = _login(client)
    changed = client.post(
        "/admin/api/users/me/change-password",
        json={"current_password": ADMIN_PASSWORD, "new_password": "rotated-admin-password"},
        headers=_bearer(session["access_token"]),
    )
    assert changed.status_code == 200, changed.text
    for old in (session, other):
        assert (
            client.get("/admin/api/auth/me", headers=_bearer(old["access_token"])).status_code
            == 401
        )
        refresh = client.post(
            "/admin/api/auth/refresh", json={"refresh_token": old["refresh_token"]}
        )
        assert refresh.status_code == 401
    # Signing straight back in (same second) works.
    fresh = _login(client, password="rotated-admin-password")
    assert client.get("/admin/api/auth/me", headers=_bearer(fresh["access_token"])).is_success


def test_role_change_revokes_the_users_tokens(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    created = client.post(
        "/admin/api/users",
        json={"email": "ops@example.com", "password": "ops-password-123", "role": "admin"},
        headers=admin_headers,
    ).json()
    ops = _login(client, "ops@example.com", "ops-password-123")
    client.patch(
        f"/admin/api/users/{created['id']}", json={"role": "viewer"}, headers=admin_headers
    )
    assert client.get("/admin/api/auth/me", headers=_bearer(ops["access_token"])).status_code == 401
    again = _login(client, "ops@example.com", "ops-password-123")
    assert again["user"]["role"] == "viewer"
