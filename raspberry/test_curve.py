"""Prova della curvatura dei bordi (stile «linee») su maschere sintetiche in coordinate del fotogramma.
Uso:  python3 test_curve.py   (servono solo numpy e opencv)"""
import sys
import numpy as np
from agrivision_rpi import nucleo as N
from agrivision_rpi.linee import curve_bordi

W, H = 960, 540
yref = int(0.72 * H)


def maschera(k0, rumore=0.0, seed=1):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), np.uint8)
    for y in range(120, H):
        t = (y - 120) / (H - 121)
        a = 400 - 210 * t + k0 * (y - yref) ** 2 + rng.normal(0, rumore)
        b = 560 + 210 * t + k0 * (y - yref) ** 2 + rng.normal(0, rumore)
        m[y, int(max(0, a)):int(min(W - 1, b)) + 1] = 1
    return m


def stima(k0, rumore=0.0):
    m = maschera(k0, rumore)
    ys = np.flatnonzero(m.any(1))
    # retta «del notebook»: minimi quadrati sui bordi, come geometria_corridoio
    sx = np.argmax(m, axis=1); dx = W - 1 - np.argmax(m[:, ::-1], axis=1)
    pl = np.polyfit(ys, sx[ys], 1); pr = np.polyfit(ys, dx[ys], 1)
    return curve_bordi(m, (pl, pr), yref, ys.min(), ys.max())


falliti = 0
def prova(nome, ok, info=""):
    global falliti
    print(("✓" if ok else "✗"), nome, info)
    falliti += (not ok)

c = stima(0.0)
prova("corridoio dritto → curvatura ~0", abs(c[0][2]) < 3e-5 and abs(c[1][2]) < 3e-5, f"({c[0][2]:.1e}, {c[1][2]:.1e})")
v = 0.0011                                                    # px/px² nel fotogramma 960 px
for k0, nome in ((v, "da un lato"), (-v, "dall'altro lato")):
    c = stima(k0)
    prova(f"sentiero che piega {nome} → curvatura vicina al vero (±25%)",
          abs(c[0][2] - k0) < 0.25 * v and abs(c[1][2] - k0) < 0.25 * v, f"({c[0][2]:+.5f}, {c[1][2]:+.5f}; vera {k0:+.5f})")
c = stima(v, rumore=4.0)
prova("con rumore sulla maschera la curvatura resta riconoscibile", abs(c[0][2] - v) < 0.4 * v and abs(c[1][2] - v) < 0.4 * v,
      f"({c[0][2]:+.5f}, {c[1][2]:+.5f})")
print("\ntutte le prove passano" if not falliti else f"\n{falliti} prove fallite")
sys.exit(1 if falliti else 0)
