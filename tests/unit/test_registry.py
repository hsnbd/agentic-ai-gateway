"""Tests for model-catalogue loading and deployment identity.

The multi-deployment cases matter more than they look: declaring several
deployments under one `model_name` is how load balancing and provider failover
are configured, so if id generation rejects or collapses them, both features
silently stop working.
"""

from __future__ import annotations

import textwrap

import pytest

from app.config.settings import Settings
from app.core.errors import ConfigurationError
from app.providers.registry import ProviderRegistry


def _registry(tmp_path, body: str) -> ProviderRegistry:
    path = tmp_path / "models.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    registry = ProviderRegistry(Settings(_env_file=None))
    registry._register_providers(None)  # type: ignore[arg-type]
    registry.load_config(str(path))
    return registry


def test_multiple_deployments_of_one_model_all_load(tmp_path):
    registry = _registry(
        tmp_path,
        """
        model_list:
          - model_name: chat
            params: {provider: openai, model: gpt-4o-mini, base_url: http://a}
          - model_name: chat
            params: {provider: openai, model: gpt-4o-mini, base_url: http://b}
          - model_name: chat
            params: {provider: openai, model: gpt-4o-mini, base_url: http://c}
        """,
    )

    deployments = registry.deployments_for("chat")
    assert len(deployments) == 3, "every deployment must survive id generation"

    ids = [d.id for d in deployments]
    assert len(set(ids)) == 3, "ids must stay distinct or routing cannot tell them apart"
    assert ids[0] == "openai/chat"
    assert ids[1:] == ["openai/chat#2", "openai/chat#3"]

    # Config order is preserved, so priority/weight stay attached to the right one.
    assert [d.base_url for d in deployments] == ["http://a", "http://b", "http://c"]


def test_explicit_duplicate_id_is_still_a_hard_error(tmp_path):
    """A hand-written duplicate id is a config mistake, not a fallback chain."""
    with pytest.raises(ConfigurationError, match="Duplicate deployment id"):
        _registry(
            tmp_path,
            """
            model_list:
              - model_name: chat
                id: shared
                params: {provider: openai, model: gpt-4o-mini}
              - model_name: other
                id: shared
                params: {provider: openai, model: gpt-4o}
            """,
        )


def test_explicit_ids_are_never_rewritten(tmp_path):
    registry = _registry(
        tmp_path,
        """
        model_list:
          - model_name: chat
            id: primary
            params: {provider: openai, model: gpt-4o-mini}
          - model_name: chat
            id: secondary
            params: {provider: openai, model: gpt-4o-mini}
        """,
    )

    assert [d.id for d in registry.deployments_for("chat")] == ["primary", "secondary"]
