"""Provider and deployment registry.

Loads `config/models.yaml`, resolves environment placeholders, and answers
"which deployments can serve this model name?" for the router.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx
import yaml

from app.config.settings import Settings
from app.core.errors import ConfigurationError, NotFoundError
from app.providers.base import Capabilities, Deployment, Pricing, Provider

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _unique_id(base: str, taken: dict[str, Any]) -> str:
    """Suffix a generated deployment id until it no longer collides.

    Several deployments of one model is the normal way to configure load
    balancing and failover, so the generated `provider/model` id is expected to
    repeat. Ids stay stable for a given config file because they are assigned in
    document order.
    """
    index = 2
    while f"{base}#{index}" in taken:
        index += 1
    return f"{base}#{index}"


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in config values."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class ProviderRegistry:
    """Owns provider adapters, deployments, and the shared HTTP client."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[str, Provider] = {}
        self._deployments: dict[str, Deployment] = {}
        #: public model name -> deployment ids, preserving config order
        self._by_model: dict[str, list[str]] = {}
        self._aliases: dict[str, str] = {}
        self._client: httpx.AsyncClient | None = None

    # -- Lifecycle --------------------------------------------------------

    async def startup(self) -> None:
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        timeout = httpx.Timeout(
            self._settings.request_timeout_seconds,
            connect=self._settings.connect_timeout_seconds,
        )
        self._client = httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=True)
        self._register_providers(self._client)
        self.load_config(self._settings.models_config_path)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _register_providers(self, client: httpx.AsyncClient) -> None:
        # Imported lazily so provider modules can import the registry types.
        from app.providers.anthropic import AnthropicProvider
        from app.providers.gemini import GeminiProvider
        from app.providers.ollama import OllamaProvider
        from app.providers.openai import OpenAIProvider

        for cls in (OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider):
            provider = cls(client)
            self._providers[provider.name] = provider

    # -- Config -----------------------------------------------------------

    def load_config(self, path: str) -> None:
        """Load (or hot-reload) deployments from a YAML file."""
        if not os.path.exists(path):
            raise ConfigurationError(f"Model config not found: {path}")

        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        raw = _expand_env(raw)
        deployments: dict[str, Deployment] = {}
        by_model: dict[str, list[str]] = {}

        for entry in raw.get("model_list") or []:
            deployment = self._build_deployment(entry)
            if deployment.id in deployments:
                # An explicit `id` colliding is always a config mistake, so it stays
                # fatal. A generated one colliding is not: declaring several
                # deployments of the same model is exactly how load balancing and
                # failover are configured, so auto-disambiguate instead of refusing
                # to start.
                if entry.get("id"):
                    raise ConfigurationError(f"Duplicate deployment id: {deployment.id}")
                deployment = deployment.model_copy(
                    update={"id": _unique_id(deployment.id, deployments)}
                )
            deployments[deployment.id] = deployment
            by_model.setdefault(deployment.model_name, []).append(deployment.id)

        self._deployments = deployments
        self._by_model = by_model
        self._aliases = dict(raw.get("aliases") or {})

    def _build_deployment(self, entry: dict[str, Any]) -> Deployment:
        model_name = entry.get("model_name")
        params = entry.get("params") or {}
        provider = params.get("provider")

        if not model_name or not provider:
            raise ConfigurationError(
                f"Each model_list entry needs model_name and params.provider: {entry!r}"
            )
        if provider not in self._providers:
            known = ", ".join(sorted(self._providers))
            raise ConfigurationError(
                f"Unknown provider {provider!r} for {model_name!r}. Known: {known}"
            )

        api_key = params.get("api_key") or self._default_api_key(provider)
        base_url = params.get("base_url") or self._default_base_url(provider)

        return Deployment(
            id=entry.get("id") or f"{provider}/{model_name}",
            model_name=model_name,
            provider=provider,
            provider_model=params.get("model") or model_name,
            api_key=api_key or None,
            base_url=base_url,
            api_version=params.get("api_version"),
            extra_headers=params.get("extra_headers") or {},
            default_params=params.get("defaults") or {},
            capabilities=Capabilities(**(entry.get("capabilities") or {})),
            pricing=Pricing(**(entry.get("pricing") or {})),
            weight=int(entry.get("weight", 1)),
            priority=int(entry.get("priority", 0)),
            rpm_limit=entry.get("rpm_limit"),
            tpm_limit=entry.get("tpm_limit"),
            enabled=bool(entry.get("enabled", True)),
            tags=entry.get("tags") or [],
        )

    def _default_api_key(self, provider: str) -> str | None:
        secret = {
            "openai": self._settings.openai_api_key,
            "anthropic": self._settings.anthropic_api_key,
            "gemini": self._settings.gemini_api_key,
        }.get(provider)
        return secret.get_secret_value() if secret else None

    def _default_base_url(self, provider: str) -> str:
        return {
            "openai": self._settings.openai_base_url,
            "anthropic": self._settings.anthropic_base_url,
            "gemini": self._settings.gemini_base_url,
            "ollama": self._settings.ollama_base_url,
        }.get(provider, "")

    # -- Lookup -----------------------------------------------------------

    def resolve_alias(self, model: str) -> str:
        seen: set[str] = set()
        current = model
        while current in self._aliases and current not in seen:
            seen.add(current)
            current = self._aliases[current]
        return current

    def deployments_for(self, model: str, *, include_disabled: bool = False) -> list[Deployment]:
        """All deployments serving a public model name, in config order."""
        resolved = self.resolve_alias(model)
        ids = self._by_model.get(resolved)

        if ids is None:
            # Allow addressing a specific deployment directly by its id.
            direct = self._deployments.get(resolved)
            if direct is None:
                raise NotFoundError(f"Model {model!r} is not configured on this gateway")
            ids = [direct.id]

        found = [self._deployments[i] for i in ids]
        return found if include_disabled else [d for d in found if d.enabled]

    def get_deployment(self, deployment_id: str) -> Deployment:
        deployment = self._deployments.get(deployment_id)
        if deployment is None:
            raise NotFoundError(f"Unknown deployment: {deployment_id}")
        return deployment

    def get_provider(self, name: str) -> Provider:
        provider = self._providers.get(name)
        if provider is None:
            raise ConfigurationError(f"Unknown provider: {name}")
        return provider

    def provider_for(self, deployment: Deployment) -> Provider:
        return self.get_provider(deployment.provider)

    def list_models(self) -> list[str]:
        return sorted(set(self._by_model) | set(self._aliases))

    def list_deployments(self) -> list[Deployment]:
        return list(self._deployments.values())

    @property
    def provider_names(self) -> list[str]:
        return sorted(self._providers)
