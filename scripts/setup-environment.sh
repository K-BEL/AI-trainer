#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION_FILE=".python-version"
DEFAULT_PYTHON_VERSION="3.12"
if [[ -f "$PYTHON_VERSION_FILE" ]]; then
  PYTHON_VERSION="$(tr -d '[:space:]' < "$PYTHON_VERSION_FILE")"
else
  PYTHON_VERSION="$DEFAULT_PYTHON_VERSION"
fi

if [[ -z "$PYTHON_VERSION" ]]; then
  PYTHON_VERSION="$DEFAULT_PYTHON_VERSION"
fi

echo "==> Project root: $ROOT_DIR"
echo "==> Target Python version: $PYTHON_VERSION"

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [[ -d "$HOME/.local/bin" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv installation failed or uv is not in PATH."
  echo "Add ~/.local/bin to PATH and run this script again."
  exit 1
fi

echo "==> Installing Python $PYTHON_VERSION with uv..."
uv python install "$PYTHON_VERSION"

echo "==> Creating/updating virtual environment..."
uv venv --python "$PYTHON_VERSION" --clear --seed .venv

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"

if ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "==> Bootstrapping pip in virtual environment..."
  "$VENV_PYTHON" -m ensurepip --upgrade
fi

echo "==> Syncing project dependencies..."
uv sync --python "$VENV_PYTHON"

echo "==> Installing package in editable mode..."
"$VENV_PYTHON" -m pip install -e .

COMPLETION_LINE='eval "$(_SIIN_TRAINER_COMPLETE=bash_source siin-trainer)"'
if [[ -f "$HOME/.bashrc" ]]; then
  if ! grep -Fq "$COMPLETION_LINE" "$HOME/.bashrc"; then
    echo "$COMPLETION_LINE" >> "$HOME/.bashrc"
    echo "==> Added CLI autocompletion to ~/.bashrc"
  fi
fi

echo
echo "Setup complete."
echo "Activate env: source $VENV_ACTIVATE"
echo "Check CLI: siin-trainer --help"
echo "Optional: wandb login"
