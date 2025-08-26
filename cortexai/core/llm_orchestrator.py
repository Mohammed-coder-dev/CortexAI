"""Light‑weight LLM orchestration layer.

The original project shipped a 20k line monolith.  This module contains a
minimal, self contained version of the orchestration logic that can be used by
both the CLI and future GUI clients.  It demonstrates how the monolith can be
split into composable modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMProvider(Enum):
    """Supported language model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMRequest:
    """Parameters used for an LLM generation request.

    A minimal dataclass replacement for the original Pydantic model.  Validation
    is performed in :func:`__post_init__` to keep external dependencies to a
    minimum.
    """

    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    stop_sequences: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")


@dataclass
class LLMResponse:
    """Result returned from the orchestrator."""

    content: str
    model: str
    latency: float
    tokens_used: int
    finish_reason: str
    is_fallback: bool = False


class LLMOrchestrator:
    """Simple orchestrator that delegates requests to provider implementations."""

    def __init__(self, providers: Dict[LLMProvider, Any]):
        self._providers = providers
        self._log = logging.getLogger(__name__)

    def _choose_provider(self, preferred: Optional[LLMProvider]) -> LLMProvider:
        if preferred and preferred in self._providers:
            return preferred
        if not self._providers:
            raise RuntimeError("no LLM providers configured")
        # Use the first configured provider as default
        return next(iter(self._providers))

    def execute(
        self,
        request: LLMRequest,
        *,
        preferred_provider: Optional[LLMProvider] = None,
    ) -> LLMResponse:
        """Execute a request against one of the configured providers."""

        provider_key = self._choose_provider(preferred_provider)
        provider = self._providers[provider_key]
        self._log.debug("selected provider %s", provider_key.value)

        content = provider.generate(request)
        return LLMResponse(
            content=content,
            model=request.model,
            latency=0.0,
            tokens_used=len(content.split()),
            finish_reason="stop",
        )
