"""Dove va il video col cruscotto: pagina web (MJPEG), finestra, file mp4, registro json."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGINA = b"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Kydo AgriVision</title><style>html,body{margin:0;height:100%;background:#000}
img{width:100%;height:100%;object-fit:contain}</style><img src="/video" alt="AgriVision">"""


class Stream:
    """Server MJPEG: apri http://<ip-della-raspberry>:<porta>/ da telefono o PC sulla stessa rete."""
    def __init__(self, porta, qualita=75):
        self.qualita, self._jpg, self._cond = qualita, None, threading.Condition()
        self._rec = {}
        padre = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
                    self.wfile.write(PAGINA)
                elif self.path == "/stato":
                    b = json.dumps(padre._rec).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(b)
                elif self.path.startswith("/video"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache"); self.end_headers()
                    try:
                        while True:
                            with padre._cond:
                                padre._cond.wait(5)
                                jpg = padre._jpg
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                            self.wfile.write(jpg); self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)
        self.server = ThreadingHTTPServer(("0.0.0.0", porta), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def pubblica(self, frame, rec=None):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.qualita])
        if ok:
            with self._cond:
                self._jpg = buf.tobytes(); self._rec = rec or {}
                self._cond.notify_all()

    def chiudi(self):
        self.server.shutdown()


class Registratore:
    def __init__(self, percorso, w, h, fps):
        self.percorso = percorso
        self.out = cv2.VideoWriter(percorso, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not self.out.isOpened():
            raise RuntimeError(f"non posso scrivere {percorso}")

    def scrivi(self, frame):
        self.out.write(frame)

    def chiudi(self):
        self.out.release()


def scrivi_registro(percorso_base, registro, unita, corridoio_cm, sessione, sorgente):
    """Stesso formato del notebook: la chiave `scarti` con i soli valori validi e un record per fotogramma."""
    p = os.path.splitext(percorso_base)[0] + "__scarti.json"
    doc = dict(sessione=sessione, errori_misurati_su="D-20", unita=unita, corridoio_cm=corridoio_cm,
               sorgente=sorgente, piattaforma="raspberry",
               scarti=[r["misura"] for r in registro if r["misura"] is not None], registro=registro)
    with open(p, "w") as f:
        json.dump(doc, f)
    return p
