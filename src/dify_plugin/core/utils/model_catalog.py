from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

_CATALOG_PROVIDER_ERROR = object()
_PROVIDER_ERROR_MESSAGES = (
    (
        InvokeAuthorizationError,
        "Incorrect model credentials provided, please check and try again.",
    ),
    (InvokeConnectionError, "Connection Error"),
    (InvokeRateLimitError, "Rate Limit Error"),
    (InvokeServerUnavailableError, "Server Unavailable Error"),
)


def sanitize_catalog_provider_error(error: Exception) -> InvokeError:
    """Keep trusted SDK categories, never provider text, args or subclasses."""
    safe_error = InvokeError("Remote model catalog provider failed")
    for error_type, message in _PROVIDER_ERROR_MESSAGES:
        if isinstance(error, error_type):
            safe_error = error_type(message)
            break
    safe_error.__dict__["_catalog_provider_error"] = _CATALOG_PROVIDER_ERROR
    return safe_error


def is_catalog_provider_error(error: Exception) -> bool:
    """Identify runtime-sanitized schema failures without changing static routes."""
    return error.__dict__.get("_catalog_provider_error") is _CATALOG_PROVIDER_ERROR


def catalog_error_object(error: Exception) -> dict[str, object]:
    """Build a bounded wire error without inspecting untrusted exception text."""
    if isinstance(error, InvokeError):
        safe_error = sanitize_catalog_provider_error(error)
        return {
            "error_type": type(safe_error).__name__,
            "message": safe_error.description,
            "args": {"description": safe_error.description},
        }
    error_type = type(error).__name__
    if error_type not in {"ValidationError", "ValueError", "TypeError"}:
        error_type = "InvokeError"
    return {
        "error_type": error_type,
        "message": "Invalid remote model catalog request",
        "args": {},
    }
