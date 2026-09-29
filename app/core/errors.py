"""Unified error taxonomy.

Every provider failure is mapped into one of these types so that routing,
retry, and fallback logic never needs provider-specific knowledge.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    # Client-side (4xx) - not retryable, not worth a fallback
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"

    # Provider-side - retryable and/or worth failing over
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_OVERLOADED = "provider_overloaded"
    CONTENT_FILTERED = "content_filtered"

    # Gateway-side
    NO_HEALTHY_DEPLOYMENT = "no_healthy_deployment"
    ALL_PROVIDERS_FAILED = "all_providers_failed"
    CONFIGURATION_ERROR = "configuration_error"
    INTERNAL_ERROR = "internal_error"


#: Errors where retrying the *same* deployment may succeed.
RETRYABLE_CODES: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.PROVIDER_RATE_LIMIT,
        ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.PROVIDER_OVERLOADED,
        ErrorCode.PROVIDER_ERROR,
    }
)

#: Errors where trying a *different* deployment may succeed.
FALLBACKABLE_CODES: frozenset[ErrorCode] = RETRYABLE_CODES | {
    ErrorCode.CONTEXT_LENGTH_EXCEEDED,
    ErrorCode.CONTENT_FILTERED,
    ErrorCode.NO_HEALTHY_DEPLOYMENT,
}

_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.AUTHENTICATION_ERROR: 401,
    ErrorCode.PERMISSION_DENIED: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.BUDGET_EXCEEDED: 402,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.GUARDRAIL_VIOLATION: 422,
    ErrorCode.CONTEXT_LENGTH_EXCEEDED: 400,
    ErrorCode.PROVIDER_RATE_LIMIT: 429,
    ErrorCode.PROVIDER_UNAVAILABLE: 503,
    ErrorCode.PROVIDER_TIMEOUT: 504,
    ErrorCode.PROVIDER_ERROR: 502,
    ErrorCode.PROVIDER_OVERLOADED: 529,
    ErrorCode.CONTENT_FILTERED: 422,
    ErrorCode.NO_HEALTHY_DEPLOYMENT: 503,
    ErrorCode.ALL_PROVIDERS_FAILED: 502,
    ErrorCode.CONFIGURATION_ERROR: 500,
    ErrorCode.INTERNAL_ERROR: 500,
}


class GatewayError(Exception):
    """Base class for all gateway errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code or _HTTP_STATUS.get(code, 500)
        self.retry_after = retry_after
        self.details = details or {}
        self.__cause__ = cause

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE_CODES

    @property
    def fallbackable(self) -> bool:
        return self.code in FALLBACKABLE_CODES

    def to_dict(self) -> dict[str, Any]:
        """Render in the OpenAI error envelope shape clients expect."""
        error: dict[str, Any] = {
            "message": self.message,
            "type": self.code.value,
            "code": self.code.value,
        }
        if self.provider:
            error["provider"] = self.provider
        if self.model:
            error["model"] = self.model
        if self.details:
            error["details"] = self.details
        return {"error": error}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value!r}, message={self.message!r})"


class InvalidRequestError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.INVALID_REQUEST, message, **kw)


class AuthenticationError(GatewayError):
    def __init__(self, message: str = "Invalid or missing API key", **kw: Any) -> None:
        super().__init__(ErrorCode.AUTHENTICATION_ERROR, message, **kw)


class PermissionDeniedError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.PERMISSION_DENIED, message, **kw)


class NotFoundError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.NOT_FOUND, message, **kw)


class BudgetExceededError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.BUDGET_EXCEEDED, message, **kw)


class RateLimitExceededError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.RATE_LIMIT_EXCEEDED, message, **kw)


class GuardrailViolationError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.GUARDRAIL_VIOLATION, message, **kw)


class ProviderError(GatewayError):
    """A provider call failed. Carries the mapped code for retry/fallback logic."""


class NoHealthyDeploymentError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.NO_HEALTHY_DEPLOYMENT, message, **kw)


class AllProvidersFailedError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.ALL_PROVIDERS_FAILED, message, **kw)


class ConfigurationError(GatewayError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(ErrorCode.CONFIGURATION_ERROR, message, **kw)
