"""Stile «assistente di linea» (retrocamera): due curve laterali e una mezzeria tratteggiata di colore
diverso che piegano seguendo il sentiero visto dal modello, al posto del trapezio con la griglia.

È la stessa idea della versione per telefono (js/geometria.js, js/disegno.js): i bordi del corridoio,
per riga della maschera, si adattano a x(y) = c0 + c1·(y − yref) + c2·(y − yref)², partendo dalla retta
del notebook. Dove il corridoio è dritto c2 ≈ 0 e la curva coincide con la retta. Lo scarto in cm resta
quello del notebook: la curva è grafica, non una misura nuova. Il righello è quello del notebook.
"""
import cv2
import numpy as np

from . import nucleo as N

SCURO = (10, 8, 6)
_prec = {"cv": None}                      # curve del fotogramma precedente, per lisciarle (media esponenziale)


def curve_bordi(maschera, geo, yref, y_alto, y_basso, margine_bordo=0.02):
    """Adatta una parabola a ciascun bordo. Lavora in coordinate del fotogramma.

    maschera: uint8 H×W (frame); geo: ((a,b) sinistro, (a,b) destro) con x = a·y + b, sempre in coordinate del fotogramma.
    Ritorna [(c0,c1,c2) sinistro, (c0,c1,c2) destro] oppure None.
    """
    m = maschera.astype(bool)
    Hf, Wf = m.shape
    ys = np.arange(int(y_alto) + 1, int(y_basso))
    ys = ys[(ys >= 0) & (ys < Hf)]
    if ys.size < 20:
        return None
    sx = np.argmax(m[ys], axis=1).astype(float)
    dx = (Wf - 1 - np.argmax(m[ys][:, ::-1], axis=1)).astype(float)
    lim = max(2.0, margine_bordo * Wf)
    presente = m[ys].any(axis=1)
    out = []
    for bordo, p, ok in ((sx, geo[0], presente & (sx > lim)), (dx, geo[1], presente & (dx < Wf - 1 - lim))):
        if ok.sum() < 15:
            return None
        u = ys[ok].astype(float) - yref
        e = bordo[ok]
        c = np.array([p[0] * yref + p[1], p[0], 0.0])               # parte dalla retta del notebook
        for _ in range(4):
            res = e - (c[0] + c[1] * u + c[2] * u * u)
            sg = max(1.4826 * float(np.median(np.abs(res))), 1.5)   # px del fotogramma
            k = np.abs(res) < 3 * sg
            if k.sum() < 12:
                break
            c = np.linalg.lstsq(np.c_[np.ones(k.sum()), u[k], u[k] ** 2], e[k], rcond=None)[0]
        out.append(tuple(float(v) for v in c))
    return out


def _limita(c2, span, larghezza, quota=0.35):
    """Nel tratto disegnato lo scostamento quadratico non supera `quota` volte la larghezza alla riga di riferimento."""
    massimo = quota * larghezza / max(span * span, 1.0)
    return float(np.clip(c2, -massimo, massimo))


def _tratto(f, P, colore, w0, w1, salta_dispari=False, k=1.0):
    """Tinta piena e punte piatte: prima tutti i contorni scuri, poi i colori, così i giunti non si vedono."""
    n = len(P)
    for passo in (0, 1):
        for i in range(n - 1):
            if salta_dispari and i % 2:
                continue
            w = max(1, int(round((w0 + (w1 - w0) * i / max(n - 2, 1)) * k)))
            if passo == 0:
                cv2.line(f, P[i], P[i + 1], SCURO, w + max(2, int(round(3 * k))), cv2.LINE_AA)
            else:
                cv2.line(f, P[i], P[i + 1], colore, w, cv2.LINE_AA)


def disegna_corridoio_linee(f, maschera, geo, stato):
    """Stessa firma e stessi valori di ritorno di nucleo._disegna_zona: (f, cm_px, centro_corr, largh)."""
    if maschera is None:
        _prec["cv"] = None
        return f, None, None, None
    Hf, Wf = f.shape[:2]
    col = {"ok": N.C_CIANO, "incerto": N.C_ARANCIO, "assente": N.C_ROSSO_S}[stato]
    m8 = (maschera > 0).astype(np.uint8)
    bx, by, bw, bh = cv2.boundingRect(m8)
    if bw < 8 or bh < 8:
        return f, None, None, None

    # maschera grezza sotto, come velo appena percettibile (serve a vedere quando il modello va male)
    roi, mroi = f[by:by + bh, bx:bx + bw], m8[by:by + bh, bx:bx + bw]
    tinta = np.empty_like(roi)
    tinta[:] = N.BLU_MARCHIO if hasattr(N, "BLU_MARCHIO") else (128, 64, 11)
    cv2.copyTo(cv2.addWeighted(roi, 1.0 - N.MASCHERA_VELO, tinta, N.MASCHERA_VELO, 0.0), mroi, roi)

    if geo is None:
        _prec["cv"] = None
        return f, None, None, None
    pl, pr = geo
    righe = np.flatnonzero(m8.any(1))
    y_alto, y_basso = float(righe.min()), float(righe.max())

    # righello: riga e larghezza lineari come nel notebook
    yref = int(np.clip(N.RIGA_RIGHELLO * Hf, N.TESTATA_H + N._k(50), Hf - N.PIEDE_H - N._k(40)))
    bs, bd = float(np.polyval(pl, yref)), float(np.polyval(pr, yref))
    largh = bd - bs
    if largh < 20:
        return f, None, None, None

    cv_ = curve_bordi(maschera, geo, yref, y_alto, y_basso)
    if cv_ is None:                                   # maschera troppo povera: due rette
        cv_ = [(float(np.polyval(pl, yref)), float(pl[0]), 0.0), (float(np.polyval(pr, yref)), float(pr[0]), 0.0)]
    elif _prec["cv"] is not None:                     # liscia nel tempo
        cv_ = [tuple(a + 0.5 * (b - a) for a, b in zip(pa, pn)) for pa, pn in zip(_prec["cv"], cv_)]
    _prec["cv"] = cv_
    span = max(abs(y_basso + N._k(6) - yref), abs(y_alto - yref))
    cv_ = [(c0, c1, _limita(c2, span, largh)) for c0, c1, c2 in cv_]

    # campioni dal basso (vicino) verso l'alto (lontano): passo che si infittisce, come la prospettiva
    n = 44
    yb = min(Hf - 1.0, y_basso + N._k(6))
    ys = [yb + (y_alto - yb) * (1 - (1 - i / (n - 1)) ** 1.6) for i in range(n)]
    def pt(c, y):
        u = y - yref
        return c[0] + c[1] * u + c[2] * u * u
    L = [(int(round(pt(cv_[0], y))), int(round(y))) for y in ys]
    R = [(int(round(pt(cv_[1], y))), int(round(y))) for y in ys]
    M = [((a[0] + b[0]) // 2, a[1]) for a, b in zip(L, R)]

    kk = 0.7 * max(0.9, N.K_UI)              # un po' più sottile del cruscotto: le linee non devono coprire il sentiero
    # tacche trasversali, come le barre di distanza di una retrocamera
    sov = f.copy()
    for t in (0.14, 0.34, 0.54, 0.72):
        i = int(round(t * (n - 1)))
        cv2.line(sov, L[i], R[i], col, max(1, int(round(2 * kk))), cv2.LINE_AA)
    cv2.addWeighted(sov, 0.55, f, 0.45, 0, f)
    att = 1.0 if stato == "ok" else 0.6
    sov = f.copy()
    _tratto(sov, L, col, 7, 2.5, False, kk)
    _tratto(sov, R, col, 7, 2.5, False, kk)
    _tratto(sov, M, N.C_WHITE, 5, 2, True, kk)           # mezzeria tratteggiata, colore diverso
    cv2.addWeighted(sov, att, f, 1 - att, 0, f)

    centro_corr = (bs + bd) / 2.0
    cm_px = (N.CORRIDOIO_CM / largh) if N.CORRIDOIO_CM else None
    N._righello_corridoio(f, col, yref, bs, bd, centro_corr, cm_px)
    return f, cm_px, centro_corr, largh
