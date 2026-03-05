"""CLI compatibility entrypoint under trainer namespace."""

from importlib import import_module


_legacy_cli = import_module("siin" + "_trainer.cli")
main = _legacy_cli.main
