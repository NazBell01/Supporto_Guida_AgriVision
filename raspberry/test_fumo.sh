#!/usr/bin/env bash
# Prova di fumo SENZA il modello vero: ONNX finto + video di un corridoio che oscilla.
# Verifica sorgente, rete, cruscotto, video in rete, registro. Non dice nulla sulla qualità del modello.
# Servono: onnx, onnxruntime, numpy, opencv, ffmpeg.  Uso: ./test_fumo.sh  [python]
set -euo pipefail
PY="${1:-python3}"; AQUI="$(cd "$(dirname "$0")" && pwd)"; T="$(mktemp -d)"; cd "$AQUI"
( cd "$T" && "$PY" "$AQUI/../test/modello_finto.py" >/dev/null && "$PY" "$AQUI/../test/video_finto.py" )
mv "$T/guida_finto.onnx" "$T/guida.onnx" 2>/dev/null || true
"$PY" -m agrivision_rpi --modello "$T/guida.onnx" --sorgente "$T/prova.mp4" --registra "$T/u.mp4" --silenzioso
"$PY" - "$T" <<'PY'
import json, os, sys
t = sys.argv[1]; d = json.load(open(f"{t}/u__scarti.json")); v = d["scarti"]
assert len(d["registro"]) > 50 and os.path.getsize(f"{t}/u.mp4") > 10000
assert min(v) < -15 and max(v) > 15, "lo scarto doveva oscillare da un lato all'altro"
print(f"OK: {len(d['registro'])} fotogrammi, scarto {min(v):+.1f} … {max(v):+.1f} cm")
PY
rm -rf "$T"
