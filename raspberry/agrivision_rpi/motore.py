"""Un fotogramma -> misura + cruscotto. Stessa logica del ciclo della cella 3 del notebook."""
import time
from collections import deque

import numpy as np
import onnxruntime as ort

from . import nucleo as N
from . import linee


class Guidatore:
    def __init__(self, percorso_onnx, corridoio_cm, larghezza, thread=0, logo="", stile="linee"):
        opz = ort.SessionOptions()
        if thread:
            opz.intra_op_num_threads = thread
        opz.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sessione = ort.InferenceSession(percorso_onnx, opz, providers=["CPUExecutionProvider"])
        dim = self.sessione.get_inputs()[0].shape            # [1, 3, A, L]
        if len(dim) != 4 or dim[2:] != [N.GUIDA_DIM[1], N.GUIDA_DIM[0]]:
            raise ValueError(f"il modello attende {dim[3]}x{dim[2]}, il simulatore {N.GUIDA_DIM[0]}x{N.GUIDA_DIM[1]}")
        self.origine_logo = N.configura(self.sessione, corridoio_cm, larghezza, logo)
        self.corridoio_cm = corridoio_cm
        # stile della guida: «linee» (assistente di linea, curve) o «area» (trapezio e griglia del notebook)
        self.disegna_zona = linee.disegna_corridoio_linee if stile == "linee" else N.disegna_corridoio
        self.N = N

        self.fps_proc = 8.0                                   # stima iniziale, poi misurata
        self._ult = None
        self._memoria_fps = None
        self._ricostruisci_storie()
        self.storia_larghezza_lunga = deque(maxlen=9000)
        self.storia_larghezza = deque(maxlen=45)
        self.registro = []
        self.n = 0
        self.ms_rete = 0.0
        self.t0 = time.time()

        # riscaldamento: il primo run di onnxruntime è sempre più lento
        x = np.zeros((1, 3, N.GUIDA_DIM[1], N.GUIDA_DIM[0]), np.float32)
        for _ in range(2):
            self.sessione.run(None, {N.NOME_IN_GUIDA: x})

    # La traccia dura SECONDI_TRACCIA secondi veri: in diretta i fotogrammi al secondo non sono
    # fissi, quindi la lunghezza della memoria segue gli fps misurati.
    def _ricostruisci_storie(self):
        n = max(30, int(round(N.SECONDI_TRACCIA * self.fps_proc)))
        vecchie = (list(getattr(self, "storia_scarto", [])), list(getattr(self, "storia_stati", [])))
        self.storia_scarto = deque(vecchie[0][-n:], maxlen=n)
        self.storia_stati = deque(vecchie[1][-n:], maxlen=n)
        self._memoria_fps = self.fps_proc

    def elabora(self, frame):
        """Disegna il cruscotto SUL fotogramma (che deve essere scrivibile) e ne restituisce il record."""
        self.n += 1
        ora = time.time()
        if self._ult is not None and ora > self._ult:
            self.fps_proc = 0.9 * self.fps_proc + 0.1 / (ora - self._ult) if self.n > 3 else 1.0 / (ora - self._ult)
            if abs(self.fps_proc - self._memoria_fps) / self._memoria_fps > 0.2:
                self._ricostruisci_storie()
        self._ult = ora

        t = time.time()
        maschera, scarto_cm, scarto, affidabile, geo, scarto_px = N.guida(frame)
        self.ms_rete = 0.9 * self.ms_rete + 0.1 * (time.time() - t) * 1000 if self.n > 1 else (time.time() - t) * 1000

        misura = scarto_cm if N.CORRIDOIO_CM is not None else scarto_px
        stato = ("ok" if misura is not None
                 else ("incerto" if maschera is not None and not affidabile else "assente"))
        self.storia_scarto.append(misura)
        self.storia_stati.append(stato)
        rec = dict(i=self.n, t=round(ora - self.t0, 3),
                   misura=None if misura is None else round(float(misura), 2), stato=stato)
        self.registro.append(rec)

        frame, cm_px, centro_corr_x, largh_px = self.disegna_zona(frame, maschera, geo, stato)

        # affidabilità geometrica e stima della larghezza, come nel notebook
        if largh_px is None:
            aff = 0.0
        else:
            med = float(np.median(self.storia_larghezza)) if len(self.storia_larghezza) >= 8 else None
            aff = 0.0 if med is None else float(np.clip(1.0 - abs(largh_px - med) / max(med, 1e-6) / 0.5, 0, 1))
            self.storia_larghezza.append(float(largh_px))
            self.storia_larghezza_lunga.append(float(largh_px))
        stima_cm = None
        if N.CORRIDOIO_CM and largh_px is not None and len(self.storia_larghezza_lunga) >= 30:
            stima_cm = N.CORRIDOIO_CM * float(np.median(list(self.storia_larghezza)[-15:])) \
                / max(float(np.median(self.storia_larghezza_lunga)), 1e-6)
        N.disegna_cruscotto_strumentale(frame, dict(
            scarto_cm=misura, larghezza_cm=N.CORRIDOIO_CM, stato=stato,
            storia=self.storia_scarto, storia_stati=self.storia_stati, fps=self.fps_proc,
            versione=N.GUIDA_SESSIONE, affidabilita=aff, stima_cm=stima_cm, cm_px=cm_px,
            centro_corr_x=centro_corr_x, unita="cm" if N.CORRIDOIO_CM is not None else "px"))
        rec["fps"] = round(self.fps_proc, 1)
        rec["ms_rete"] = round(self.ms_rete, 1)
        return frame, rec
