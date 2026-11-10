#!/usr/bin/env bash
# One-time setup for tools/asr/speech_wer.py: a Python env with faster-whisper in
# ~/.cache/chimera-asr (the small.en model, ~0.5 GB, downloads there on first use).
set -euo pipefail
dir="$HOME/.cache/chimera-asr"
mkdir -p "$dir/models"
[ -x "$dir/.venv/bin/python" ] || uv venv -q --python 3.12 "$dir/.venv"
uv pip install -q --python "$dir/.venv/bin/python" faster-whisper==1.2.1
echo "run: $dir/.venv/bin/python tools/asr/speech_wer.py <outdir> [--against earlier/wer.json]"
