"""Prove dello stile «linee» (nastro: linea centrale che flette + semilarghezza in prospettiva).
  1. fedeltà: su sentieri puliti (dritto, curvo da un lato, dall'altro, con rumore) il nastro ritrova la curvatura vera
     e i bordi disegnati cadono sui bordi della maschera;
  2. robustezza: su maschere SPORCHE (morsi di fogliame, buchi, rumore) le due curve non si incrociano mai.
Uso:  PYTHONPATH=. python3 test_curve.py   (servono solo numpy e opencv)"""
import sys
import numpy as np
from agrivision_rpi.linee import nastro_bordi, punti_nastro, estremo_lontano

W, H = 960, 540
yref = int(0.72 * H)
falliti = 0


def prova(nome, ok, info=""):
    global falliti
    print(("✓" if ok else "✗"), nome, info)
    falliti += (not ok)


def retta_notebook(m):
    ys = np.flatnonzero(m.any(1)); ys = ys[(ys > ys.min() + 0.05 * H) & (ys < 0.97 * H)]
    sx = np.argmax(m, axis=1); dx = W - 1 - np.argmax(m[:, ::-1], axis=1)
    ls = ys[sx[ys] > 19]; ld = ys[dx[ys] < W - 20]
    if len(ls) < 10 or len(ld) < 10:
        return None
    return np.polyfit(ls, sx[ls], 1), np.polyfit(ld, dx[ld], 1)


# ---- 1. fedeltà su sentieri puliti ------------------------------------------------------------
def pulita(k0, rumore=0.0, seed=1):
    rng = np.random.default_rng(seed); m = np.zeros((H, W), np.uint8)
    for y in range(120, H):
        t = (y - 120) / (H - 121)
        c = 480 + k0 * (y - yref) ** 2; h = 70 + 210 * t
        m[y, int(max(0, c - h + rng.normal(0, rumore))):int(min(W - 1, c + h + rng.normal(0, rumore))) + 1] = 1
    return m


def stima(m):
    geo = retta_notebook(m); ys = np.flatnonzero(m.any(1))
    par = nastro_bordi(m, geo, yref, ys.min(), ys.max())
    return par, geo, ys


v = 0.0011
for k0, nome in ((0.0, "dritto"), (v, "da un lato"), (-v, "dall'altro lato")):
    m = pulita(k0); par, geo, ys = stima(m)
    c2 = par[0][2]
    righe = np.arange(ys.min() + 40, ys.max() - 5, 10)
    xl, xr, _ = punti_nastro(par, geo, righe, yref)
    vero_l = np.argmax(m[righe], axis=1); vero_r = W - 1 - np.argmax(m[righe][:, ::-1], axis=1)
    err = max(np.abs(xl - vero_l).max(), np.abs(xr - vero_r).max())
    ok = (abs(c2) < 3e-5) if k0 == 0 else (abs(c2 - k0) < 0.3 * v)
    prova(f"sentiero {nome}: curvatura {'~0' if k0 == 0 else 'vicina al vero'} e bordi sulla maschera",
          ok and err < 14, f"(c2 {c2:+.5f}, vera {k0:+.5f}; errore massimo sui bordi {err:.1f} px su 960)")
m = pulita(v, rumore=4.0); par, geo, ys = stima(m)
prova("con rumore la curvatura resta riconoscibile", abs(par[0][2] - v) < 0.45 * v, f"({par[0][2]:+.5f})")


# ---- 2. robustezza su maschere sporche ----------------------------------------------------------
def sporca(rng):
    m = np.zeros((H, W), np.uint8)
    yaw = rng.uniform(-120, 120); top = int(rng.uniform(130, 200))
    for y in range(top, H):
        t = (y - top) / (H - top); c = 480 + yaw * (1 - t) * 0.6; h = 60 + 300 * t
        m[y, int(max(0, c - h + rng.normal(0, 3))):int(min(W - 1, c + h + rng.normal(0, 3))) + 1] = 1
    for _ in range(rng.integers(1, 5)):                           # morsi di fogliame da un lato
        y0 = int(rng.uniform(top, H - 80)); hh = int(rng.uniform(40, 220)); prof = int(rng.uniform(60, 320))
        if rng.random() < 0.5: m[y0:y0 + hh, :int(480 + yaw - 60) + prof] = 0
        else: m[y0:y0 + hh, int(480 + yaw + 60) - prof:] = 0
    for _ in range(rng.integers(0, 30)):
        x, y = rng.integers(100, 860), rng.integers(top, H - 20); m[y:y + 12, x:x + 20] = 0
    return m


rng = np.random.default_rng(3); n = incroci = 0; minw = 1e9
for _ in range(600):
    m = sporca(rng); geo = retta_notebook(m)
    if geo is None: continue
    ys = np.flatnonzero(m.any(1)); par = nastro_bordi(m, geo, yref, ys.min(), ys.max())
    if par is None: continue
    n += 1
    xl, xr, _ = punti_nastro(par, geo, np.linspace(ys.min(), ys.max() + 6, 60), yref)
    w = xr - xl; minw = min(minw, w.min()); incroci += int((w <= 0).any())
prova("600 maschere sporche: le curve non si incrociano mai", n > 300 and incroci == 0, f"({n} valutate, incroci {incroci}, larghezza minima {minw:.0f} px)")



# ---- 3. nessuno spigolo: la linea non «si impenna» in lontananza ------------------------------------
def svolta_max(xs, ys):
    P = np.c_[xs, ys]; d = np.diff(P, axis=0); a = np.arctan2(d[:, 1], d[:, 0])
    return float(np.degrees(np.abs(np.angle(np.exp(1j * np.diff(a))))).max())


rng = np.random.default_rng(3); n = spigoli = 0; media = 0.0
for _ in range(600):
    m = sporca(rng); geo = retta_notebook(m)
    if geo is None: continue
    ys = np.flatnonzero(m.any(1)); par = nastro_bordi(m, geo, yref, ys.min(), ys.max())
    if par is None: continue
    n += 1
    ya = estremo_lontano(geo, yref, ys.min(), ys.max())                 # lo stesso taglio che usa il disegno
    campioni = np.array([ys.max() + 6 + (ya - ys.max() - 6) * (1 - (1 - i / 43) ** 1.6) for i in range(44)])
    xl, xr, xm = punti_nastro(par, geo, campioni, yref)
    mx = max(svolta_max(xl, campioni), svolta_max(xr, campioni), svolta_max(xm, campioni))
    media += mx; spigoli += int(mx > 25)
prova("600 maschere sporche: nessuna linea con uno spigolo oltre 25°", n > 300 and spigoli == 0,
      f"({n} valutate, con spigolo {spigoli}, svolta massima media {media / n:.1f}°)")

print("\ntutte le prove passano" if not falliti else f"\n{falliti} prove fallite")
sys.exit(1 if falliti else 0)
