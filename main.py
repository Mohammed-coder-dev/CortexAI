"""Simplified entry point for CortexAI.

The original repository bundled every component of the system into a single
20k line script.  This refactored entry point imports a handful of modules from
the new :mod:`cortexai` package and wires them together.  It serves as an
example of how the project can evolve towards a maintainable architecture.
"""

from __future__ import annotations

from cortexai.core.llm_orchestrator import (
    LLMOrchestrator,
    LLMProvider,
    LLMRequest,
)
from cortexai.ui.error_dialog import error_handler


class EchoProvider:
    """Minimal provider used for demonstration and testing."""

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - trivial
        return f"echo: {request.prompt}"


@error_handler
def main() -> None:  # pragma: no cover - thin wrapper
    orchestrator = LLMOrchestrator({LLMProvider.LOCAL: EchoProvider()})
    req = LLMRequest(prompt="Hello CortexAI", model="demo")
    resp = orchestrator.execute(req)
    print(resp.content)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
