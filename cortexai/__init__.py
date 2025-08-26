"""CortexAI refactored package.

Only a tiny subset of the original project is included here.  The goal of this
package is to showcase a maintainable, modular structure that can be expanded in
future iterations.
"""

from .core.llm_orchestrator import LLMOrchestrator, LLMProvider, LLMRequest, LLMResponse
from .ui.error_dialog import error_handler

__all__ = [
    "LLMOrchestrator",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "error_handler",
]
