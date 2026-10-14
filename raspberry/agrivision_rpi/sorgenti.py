"""Sorgenti video: webcam USB, videocamera Raspberry (CSI, picamera2), file o flusso di rete.

Un thread legge di continuo e tiene SOLO l'ultimo fotogramma: se la rete è più lenta della
videocamera i fotogrammi vecchi si scartano invece di accumularsi (altrimenti il ritardo cresce
e il cruscotto mostra il passato). Con un file, invece, si legge fotogramma per fotogramma.
"""
import threading
import time

import cv2


class Sorgente:
    def __init__(self, spec, larghezza, altezza, fps=30):
        self.spec = spec
        self.ultimo = None
        self.numero = 0
        self._stop = False
        self._lock = threading.Lock()
        self.da_file = False
        self.fps_nativi = fps
        self._apri(spec, larghezza, altezza, fps)
        if not self.da_file:
            self._th = threading.Thread(target=self._ciclo, daemon=True)
            self._th.start()

    def _apri(self, spec, w, h, fps):
        if spec in ("picamera", "csi"):
            from picamera2 import Picamera2                  # solo sulla Raspberry
            self.cam = Picamera2()
            # "RGB888" in picamera2 produce byte in ordine B,G,R: è già l'ordine di OpenCV
            conf = self.cam.create_video_configuration(
                main={"size": (w, h), "format": "RGB888"}, controls={"FrameRate": fps})
            self.cam.configure(conf)
            self.cam.start()
            self._leggi = lambda: (True, self.cam.capture_array())
            self._chiudi = lambda: (self.cam.stop(), self.cam.close())
            self.descr = f"videocamera Raspberry {w}x{h}"
        elif spec.startswith("usb"):
            idx = int(spec.split(":")[1]) if ":" in spec else 0
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # molte webcam danno 720p solo in MJPG
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f"webcam /dev/video{idx} non apribile (elenco: v4l2-ctl --list-devices)")
            self._leggi = cap.read
            self._chiudi = cap.release
            self.descr = f"webcam USB /dev/video{idx} {int(cap.get(3))}x{int(cap.get(4))}"
        else:                                                # file o URL (rtsp://, http://)
            cap = cv2.VideoCapture(spec)
            if not cap.isOpened():
                raise RuntimeError(f"non riesco ad aprire «{spec}»")
            self.da_file = not spec.startswith(("rtsp://", "http://", "https://"))
            self.fps_nativi = cap.get(cv2.CAP_PROP_FPS) or fps
            self._leggi = cap.read
            self._chiudi = cap.release
            self.descr = f"{'file' if self.da_file else 'flusso'} {spec}"

    def _ciclo(self):
        while not self._stop:
            ok, f = self._leggi()
            if not ok:
                time.sleep(0.05); continue
            with self._lock:
                self.ultimo, self.numero = f, self.numero + 1

    def leggi(self, dopo=0, attesa=5.0):
        """Ritorna (ok, fotogramma, numero). Da videocamera: l'ultimo più recente di `dopo`."""
        if self.da_file:
            ok, f = self._leggi()
            self.numero += 1
            return ok, f, self.numero
        t = time.time()
        while time.time() - t < attesa:
            with self._lock:
                if self.ultimo is not None and self.numero > dopo:
                    return True, self.ultimo.copy(), self.numero
            time.sleep(0.003)
        return False, None, dopo

    def chiudi(self):
        self._stop = True
        try:
            self._chiudi()
        except Exception:
            pass
