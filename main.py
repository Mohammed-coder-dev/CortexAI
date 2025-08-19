#!/usr/bin/env python3
"""CortexAI monolithic application.

This refactored version provides a compact yet extensible
single-file application that demonstrates structured error handling
and a lightweight plugin system.  It intentionally keeps the codebase
monolithic while organising the functionality into classes so that the
file remains maintainable.

Run with ``--list`` to view available plugins or ``--plugin NAME`` to
execute a plugin.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOGGER_NAME = "cortexai"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class ErrorHandler:
    """Simple context manager for catching and logging exceptions.

    Any unhandled exception is logged before propagating the error.  This
    keeps the application monolithic while providing a central place for
    error related logic.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "ErrorHandler":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc:
            logger.error("%s crashed: %s", self.name, exc, exc_info=exc)
        return False  # propagate exception


# ---------------------------------------------------------------------------
# Plugin infrastructure
# ---------------------------------------------------------------------------
@dataclass
class Plugin:
    """Represents a pluggable command."""

    name: str
    run: Callable[[argparse.Namespace], None]
    description: str = ""


class PluginManager:
    """Load and execute plugins.

    Plugins are Python modules that expose ``create_plugin()`` returning a
    :class:`Plugin` instance.  Modules can be dropped into the ``plugins``
    directory or registered programmatically.
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, Plugin] = {}

    # registration -----------------------------------------------------
    def register(self, plugin: Plugin) -> None:
        logger.debug("Registering plugin %s", plugin.name)
        self._plugins[plugin.name] = plugin

    def discover(self, directory: str = "plugins") -> None:
        if not os.path.isdir(directory):
            return
        for path in os.listdir(directory):
            if not path.endswith(".py"):
                continue
            name = path[:-3]
            file_path = os.path.join(directory, path)
            spec = importlib.util.spec_from_file_location(name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "create_plugin"):
                    plugin = module.create_plugin()
                    self.register(plugin)

    # execution -------------------------------------------------------
    def names(self) -> Iterable[str]:
        return self._plugins.keys()

    def get(self, name: str) -> Optional[Plugin]:
        return self._plugins.get(name)

    def run(self, name: str, args: argparse.Namespace) -> None:
        plugin = self.get(name)
        if not plugin:
            raise ValueError(f"Unknown plugin: {name}")
        plugin.run(args)


# ---------------------------------------------------------------------------
# Built‑in example plugin
# ---------------------------------------------------------------------------

def _create_echo_plugin() -> Plugin:
    def run(args: argparse.Namespace) -> None:
        print(args.message)

    parser = argparse.ArgumentParser(prog="echo", add_help=False)
    parser.add_argument("message", help="Message to echo")

    def wrapper(namespace: argparse.Namespace) -> None:
        run(parser.parse_args(namespace.plugin_args))

    return Plugin(name="echo", run=wrapper, description="Echo back a message")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class CortexAIApp:
    """Command line interface for executing plugins."""

    def __init__(self) -> None:
        self.plugins = PluginManager()
        self.plugins.register(_create_echo_plugin())
        self.plugins.discover()

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="CortexAI monolithic app")
        parser.add_argument("--list", action="store_true", help="List plugins and exit")
        parser.add_argument("--plugin", help="Plugin to execute")
        parser.add_argument("plugin_args", nargs=argparse.REMAINDER,
                            help="Arguments passed to the plugin")
        return parser

    def run(self, argv: Optional[Iterable[str]] = None) -> int:
        parser = self.create_parser()
        args = parser.parse_args(argv)
        if args.list:
            for name in sorted(self.plugins.names()):
                plugin = self.plugins.get(name)
                print(f"{name}: {plugin.description}")
            return 0
        if not args.plugin:
            parser.print_help()
            return 1
        args.plugin_args = args.plugin_args or []
        self.plugins.run(args.plugin, args)
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    app = CortexAIApp()
    with ErrorHandler("CortexAI"):
        return app.run(argv)


if __name__ == "__main__":
    sys.exit(main())
