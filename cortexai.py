"""CortexAI runtime module with plugin architecture and CLI interface.

This module exposes the :class:`CortexAI` class which manages a registry of
plugins.  Plugins follow a simple lifecycle of ``initialize`` → ``execute`` →
``shutdown`` and can be dynamically loaded at runtime.  A small command line
interface is provided for experimentation and integration in larger systems.

The design is intentionally lightweight but follows patterns that make the code
"market ready":

* Structured logging using the :mod:`logging` module
* Clear error handling with helpful exceptions
* Type annotations and extensive docstrings for maintainability
* No hard dependency on external packages to ease deployment
* Simple plugin discovery from ``module:ClassName`` strings
"""
from __future__ import annotations

import argparse
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Type


class PluginError(RuntimeError):
    """Raised when a plugin fails to load or execute."""


@dataclass
class BasePlugin:
    """Base class that all CortexAI plugins should inherit from."""

    name: str = "base"
    config: Dict[str, Any] = field(default_factory=dict)

    def initialize(self) -> None:
        """Hook executed when the plugin is registered."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - interface
        """Run the plugin and return a result.

        Subclasses must override this method.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """Hook executed when the runtime shuts down."""


class CortexAI:
    """Core runtime responsible for managing plugins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.plugins: Dict[str, BasePlugin] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("CortexAI initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Plugin management
    # ------------------------------------------------------------------
    def register_plugin(
        self, plugin_cls: Type[BasePlugin], *, name: Optional[str] = None, **config: Any
    ) -> None:
        """Register and initialise a plugin.

        Args:
            plugin_cls: Class implementing :class:`BasePlugin`.
            name: Optional custom name for the plugin.
            **config: Configuration passed to the plugin constructor.
        """
        plugin_name = name or getattr(plugin_cls, "name", plugin_cls.__name__)
        if plugin_name in self.plugins:
            raise PluginError(f"Plugin '{plugin_name}' already registered")

        plugin = plugin_cls(config)
        try:
            plugin.initialize()
        except Exception as exc:  # pragma: no cover - defensive
            raise PluginError(f"Failed to initialise plugin '{plugin_name}'") from exc

        self.plugins[plugin_name] = plugin
        self.logger.info("Registered plugin '%s'", plugin_name)

    def run_plugin(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered plugin and return its result."""
        if name not in self.plugins:
            raise PluginError(f"Plugin '{name}' is not registered")
        self.logger.debug("Executing plugin '%s'", name)
        plugin = self.plugins[name]
        try:
            return plugin.execute(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            raise PluginError(f"Plugin '{name}' execution failed") from exc

    def shutdown(self) -> None:
        """Shutdown all registered plugins in reverse order."""
        for name, plugin in reversed(self.plugins.items()):
            try:
                plugin.shutdown()
                self.logger.debug("Plugin '%s' shut down", name)
            except Exception:  # pragma: no cover - defensive
                self.logger.exception("Error shutting down plugin '%s'", name)


# ----------------------------------------------------------------------
# Helper functions & CLI
# ----------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for the runtime."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_plugin_from_path(path: str) -> Type[BasePlugin]:
    """Load a plugin class from a ``module:ClassName`` string."""
    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise PluginError("Plugin path must be in 'module:ClassName' format")

    module = importlib.import_module(module_name)
    plugin_cls = getattr(module, class_name, None)
    if plugin_cls is None or not issubclass(plugin_cls, BasePlugin):
        raise PluginError(f"{class_name!r} is not a valid plugin class")
    return plugin_cls


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entry point for the CortexAI command line interface."""
    parser = argparse.ArgumentParser(description="CortexAI runtime")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--plugin",
        help="Load and execute a plugin from 'module:ClassName'",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(args.log_level)
    runtime = CortexAI()

    if args.plugin:
        plugin_cls = load_plugin_from_path(args.plugin)
        runtime.register_plugin(plugin_cls)
        runtime.run_plugin(plugin_cls.name)
    return 0


if __name__ == "__main__":  # pragma: no cover - script execution
    raise SystemExit(main())
