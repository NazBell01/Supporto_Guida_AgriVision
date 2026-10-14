#!/usr/bin/env bash
# Installazione di AgriVision su Raspberry Pi OS (64 bit, Bookworm o successivo).
#   ./installa.sh              prepara l'ambiente
#   ./installa.sh servizio     in più, avvia AgriVision da solo all'accensione
set -euo pipefail
cd "$(dirname "$0")"

echo "== Pacchetti di sistema =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-opencv python3-numpy v4l-utils
# La videocamera Raspberry (CSI) vuole picamera2: facoltativo se usi solo una webcam USB.
sudo apt-get install -y python3-picamera2 || echo "(picamera2 non disponibile: userai una webcam USB o un file)"

echo "== Ambiente Python =="
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "== Controllo =="
.venv/bin/python - <<'PY'
import cv2, numpy, onnxruntime
print("opencv", cv2.__version__, "| numpy", numpy.__version__, "| onnxruntime", onnxruntime.__version__)
PY

if [[ "${1:-}" == "servizio" ]]; then
  MODELLO="${MODELLO:-$PWD/guida.onnx}"
  SORGENTE="${SORGENTE:-picamera}"
  [[ -f "$MODELLO" ]] || { echo "Metti prima il modello in $MODELLO (o usa MODELLO=/percorso/guida.onnx)"; exit 1; }
  UNIT=/etc/systemd/system/agrivision.service
  sudo tee "$UNIT" >/dev/null <<UNITA
[Unit]
Description=Kydo AgriVision (ausilio alla guida)
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$PWD
ExecStart=$PWD/.venv/bin/python -m agrivision_rpi --modello $MODELLO --sorgente $SORGENTE --stream 8080 --silenzioso --registro $PWD/registri/uscita
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNITA
  mkdir -p "$PWD/registri"
  sudo systemctl daemon-reload
  sudo systemctl enable --now agrivision.service
  echo "Servizio attivo. Stato: systemctl status agrivision · log: journalctl -u agrivision -f"
fi
echo "Fatto."
