#!/usr/bin/env python3
"""CortexAI monolithic application.

This compact yet extensible single-file app demonstrates structured
error handling and a lightweight plugin system. While monolithic by
design, the code is modularised for maintainability.

Usage:
    python cortexai.py --list
    python cortexai.py --plugin echo hello world
    python cortexai.py --plugin echo --help
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
# Config
# ---------------------------------------------------------------------------
APP_NAME = "CortexAI"
PLUGINS_DIR = "plugins"

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOGGER_NAME = "cortexai"
logger = logging.getLogger(LOGGER_NAME)


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class ErrorHandler:
    """Context manager for catching/logging exceptions."""

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
    name: str
    run: Callable[[argparse.Namespace], None]
    description: str = ""
    parser: Optional[argparse.ArgumentParser] = None


class PluginManager:
    def __init__(self) -> None:
        self._plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        logger.debug("Registering plugin %s", plugin.name)
        self._plugins[plugin.name] = plugin

    def discover(self, directory: str = PLUGINS_DIR) -> None:
        if not os.path.isdir(directory):
            logger.debug("No plugin directory found: %s", directory)
            return
        for path in os.listdir(directory):
            if not path.endswith(".py"):
                continue
            try:
                name = path[:-3]
                file_path = os.path.join(directory, path)
                spec = importlib.util.spec_from_file_location(name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "create_plugin"):
                        plugin = module.create_plugin()
                        self.register(plugin)
                        logger.info("Loaded plugin: %s", name)
            except Exception as e:
                logger.warning("Failed to load plugin %s: %s", path, e)

    def names(self) -> Iterable[str]:
        return self._plugins.keys()

    def get(self, name: str) -> Optional[Plugin]:
        return self._plugins.get(name)

    def run(self, name: str, args: argparse.Namespace) -> None:
        plugin = self.get(name)
        if not plugin:
            raise ValueError(f"Unknown plugin: {name}")
        if plugin.parser and "--help" in args.plugin_args:
            plugin.parser.print_help()
            return
        plugin.run(args)

# ---------------------------------------------------------------------------
# Built-in example plugin
# ---------------------------------------------------------------------------
def _create_echo_plugin() -> Plugin:
    parser = argparse.ArgumentParser(prog="echo", description="Echo back a message")
    parser.add_argument("message", help="Message to echo")

    def run(namespace: argparse.Namespace) -> None:
        ns = parser.parse_args(namespace.plugin_args)
        print(ns.message)

    return Plugin(name="echo", run=run, description="Echo back a message", parser=parser)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class CortexAIApp:
    def __init__(self) -> None:
        self.plugins = PluginManager()
        self.plugins.register(_create_echo_plugin())
        self.plugins.discover()

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=f"{APP_NAME} monolithic app")
        parser.add_argument("--list", action="store_true", help="List plugins and exit")
        parser.add_argument("--plugin", help="Plugin to execute")
        parser.add_argument(
            "--verbose", action="store_true", help="Enable debug logging"
        )
        parser.add_argument(
            "--quiet", action="store_true", help="Suppress info logs"
        )
        parser.add_argument(
            "plugin_args", nargs=argparse.REMAINDER, help="Arguments passed to the plugin"
        )
        return parser

    def run(self, argv: Optional[Iterable[str]] = None) -> int:
        parser = self.create_parser()
        args = parser.parse_args(argv)

        configure_logging(args.verbose, args.quiet)

        if args.list:
            for name in sorted(self.plugins.names()):
                plugin = self.plugins.get(name)
                desc = plugin.description if plugin else ""
                print(f"{name}: {desc}")
            return 0

        if not args.plugin:
            parser.print_help()
            return 2  # exit code for bad usage

        args.plugin_args = args.plugin_args or []
        try:
            self.plugins.run(args.plugin, args)
            return 0
        except ValueError as e:
            logger.error(e)
            return 3

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    app = CortexAIApp()
    with ErrorHandler(APP_NAME):
        return app.run(argv)

if __name__ == "__main__":
    sys.exit(main())
