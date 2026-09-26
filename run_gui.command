#!/bin/bash
# Hit-Sync for macOS: double-click in Finder (the first time: right-click ->
# Open, because the file was downloaded). The first run sets everything up.
cd "$(dirname "$0")" || exit 1
pause() { echo; read -r -n1 -p "Press any key to close..."; }
if [ ! -x .venv/bin/python ]; then
  echo "Setting up Hit-Sync for the first time. This takes a few minutes..."
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3.10 or newer is needed: https://www.python.org/downloads/macos/"
    echo "(or with Homebrew: brew install python ffmpeg)"
    pause; exit 1
  fi
  python3 -m venv .venv || { pause; exit 1; }
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "Install failed."; pause; exit 1; }
  cp requirements.txt .venv/requirements.installed
fi
exec .venv/bin/python -m hitsync "$@"
