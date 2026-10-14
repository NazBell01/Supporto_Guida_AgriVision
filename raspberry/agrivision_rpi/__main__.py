"""python3 -m agrivision_rpi --modello guida.onnx --sorgente picamera --stream 8080"""
import argparse
import os
import signal
import sys
import time

import cv2

from . import nucleo as N
from .motore import Guidatore
from .sorgenti import Sorgente
from .uscite import Registratore, Stream, scrivi_registro


def argomenti():
    p = argparse.ArgumentParser(prog="agrivision_rpi",
        description="AgriVision REV_R su Raspberry Pi: scarto laterale dal corridoio, in tempo reale.")
    p.add_argument("--modello", required=True, help="file guida.onnx (esportato dalla cella 1bis del notebook)")
    p.add_argument("--sorgente", default="usb:0",
                   help="usb:0 | picamera | percorso di un file video | rtsp://… (default usb:0)")
    p.add_argument("--larghezza", type=int, default=960, help="larghezza di lavoro in pixel (default 960)")
    p.add_argument("--altezza", type=int, default=0, help="altezza richiesta alla camera (default 16:9 dalla larghezza)")
    p.add_argument("--corridoio-cm", type=float, default=200.0,
                   help="larghezza reale del corridoio in cm (default 200, Vigna94); 0 = scala non misurata")
    p.add_argument("--thread", type=int, default=0, help="thread di onnxruntime (0 = automatico)")
    p.add_argument("--stream", type=int, default=0, metavar="PORTA", help="pagina web con il video, es. 8080")
    p.add_argument("--finestra", action="store_true", help="mostra una finestra (serve uno schermo collegato)")
    p.add_argument("--registra", metavar="FILE.mp4", help="salva il video col cruscotto e il registro degli scarti")
    p.add_argument("--registro", metavar="BASE", help="salva solo il registro json (senza video)")
    p.add_argument("--logo", default="", help="logo del marchio (png/jpg); se manca c'è quello incorporato")
    p.add_argument("--max-fotogrammi", type=int, default=0, help="si ferma dopo N fotogrammi (prove)")
    p.add_argument("--silenzioso", action="store_true")
    return p.parse_args()


def principale():
    a = argomenti()
    if not os.path.exists(a.modello):
        sys.exit(f"modello non trovato: {a.modello}")
    w = a.larghezza; h = a.altezza or (w * 9 // 16)
    corr = a.corridoio_cm if a.corridoio_cm > 0 else None

    src = Sorgente(a.sorgente, w, h)
    print(f"sorgente: {src.descr}")
    ok, f, _ = src.leggi()
    if not ok:
        sys.exit("nessun fotogramma dalla sorgente")
    if f.shape[1] != w:                                    # lavoro sempre alla larghezza richiesta
        h = int(round(f.shape[0] * w / f.shape[1])); h -= h % 2
    W, H = w, h
    print(f"lavoro a {W}x{H}")

    g = Guidatore(a.modello, corr, W, a.thread, a.logo)
    print(f"modello: {a.modello} · scala: {'%.0f cm' % corr if corr else 'NON misurata (pixel)'} · "
          f"marchio: {g.origine_logo}")
    print(f"rete: {g.ms_rete:.0f} ms (riscaldata)" if g.ms_rete else "")

    stream = Stream(a.stream) if a.stream else None
    if stream:
        print(f"video in rete su  http://<ip-della-raspberry>:{a.stream}/   (ip: hostname -I)")
    reg = Registratore(a.registra, W, H, 15) if a.registra else None
    if a.finestra:
        cv2.namedWindow("AgriVision", cv2.WINDOW_NORMAL)

    fermo = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: fermo.update(v=True))
    signal.signal(signal.SIGTERM, lambda *_: fermo.update(v=True))

    ultimo, n = 0, 0
    t_stampa = time.time()
    while not fermo["v"]:
        ok, frame, ultimo = src.leggi(ultimo)
        if not ok:
            if src.da_file:
                break
            print("nessun fotogramma negli ultimi 5 s: la camera si è scollegata?"); continue
        if frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        frame = frame.copy() if not frame.flags.writeable else frame
        frame, rec = g.elabora(frame)
        n += 1
        if stream: stream.pubblica(frame, rec)
        if reg: reg.scrivi(frame)
        if a.finestra:
            cv2.imshow("AgriVision", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        if not a.silenzioso and time.time() - t_stampa > 2:
            t_stampa = time.time()
            sc = "n/d" if rec["misura"] is None else f"{rec['misura']:+.1f} {g.N.CORRIDOIO_CM and 'cm' or 'px'}"
            print(f"fotogramma {n:5d} | scarto {sc:>10} | {rec['stato']:8s} | rete {rec['ms_rete']:.0f} ms | {rec['fps']:.1f} fps")
        if a.max_fotogrammi and n >= a.max_fotogrammi:
            break

    src.chiudi()
    if stream: stream.chiudi()
    if reg: reg.chiudi()
    if a.finestra: cv2.destroyAllWindows()
    base = a.registra or a.registro
    if base and g.registro:
        p = scrivi_registro(base, g.registro, "cm" if corr else "px", corr, N.GUIDA_SESSIONE, a.sorgente)
        print(f"registro scritto: {p}")
    print(f"fine: {n} fotogrammi")


if __name__ == "__main__":
    principale()
