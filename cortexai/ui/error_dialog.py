"""Utility helpers for error handling.

The original implementation used a complex PyQt5 based dialog.  The refactored
version keeps the public API small and framework agnostic so it can be reused by
both CLI and GUI front ends.
"""

from __future__ import annotations

import functools
import logging
import traceback
from typing import Any, Callable, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


def error_handler(func: T) -> T:
    """Log uncaught exceptions from *func* and re-raise them.

    This decorator preserves the call signature of the wrapped function and
    provides a single place to hook in structured logging or telemetry.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - trivial wrapper
            logging.error("Unhandled error in %s: %s", func.__name__, exc)
            logging.debug("%s", traceback.format_exc())
            raise

    return wrapper  # type: ignore[misc]


__all__ = ["error_handler"]
