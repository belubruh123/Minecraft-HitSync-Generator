#!/bin/sh
# Hit-Sync for Linux. Needs Python 3.10+; Qt and preview sound need the
# system libraries libegl1 and libportaudio2 (e.g. sudo apt install libegl1 libportaudio2).
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Setting up Hit-Sync for the first time..."
  python3 -m venv .venv || exit 1
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || exit 1
  cp requirements.txt .venv/requirements.installed
fi
exec .venv/bin/python -m hitsync "$@"
