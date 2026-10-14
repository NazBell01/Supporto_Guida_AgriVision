"""Stile «assistente di linea» (retrocamera): due curve laterali e una mezzeria tratteggiata di colore
diverso che piegano seguendo il sentiero visto dal modello, al posto del trapezio con la griglia.

Il sentiero è un NASTRO: una linea centrale che flette (parabola) e una semilarghezza in prospettiva
(retta); i bordi sono centro ∓ semilarghezza, quindi non possono incrociarsi e piegano insieme. Parametri
adattati alla maschera e vincolati alle rette del notebook (validate): larghezza fra 0,6 e 1,4 volte quella
della retta, centro entro mezza semilarghezza. È la stessa idea della versione per telefono. Lo scarto in cm
resta quello del notebook: le curve sono grafica, non una misura nuova. Il righello è quello del notebook.
"""
import cv2
import numpy as np

from . import nucleo as N

SCURO = (10, 8, 6)
SMORZ = 0.35                              # peso del fotogramma nuovo nella media esponenziale (più basso = più stabile)
_prec = {"cv": None}                      # curve del fotogramma precedente, per lisciarle (media esponenziale)


def _fit_robusto(u, e, c0, grado, sg_min=1.5, giri=4):
    """Minimi quadrati di e ≈ polinomio(u) con scarto dei punti lontani (3 sigma robusti), da una stima iniziale c0."""
    c = np.array(c0, float)
    base = lambda uu: np.vstack([uu ** k for k in range(grado + 1)]).T
    for _ in range(giri):
        res = e - base(u) @ c
        sg = max(1.4826 * float(np.median(np.abs(res))), sg_min)
        k = np.abs(res) < 3 * sg
        if k.sum() < 3 * (grado + 1):
            break
        c = np.linalg.lstsq(base(u[k]), e[k], rcond=None)[0]
    return c


def nastro_bordi(maschera, geo, yref, y_alto, y_basso, margine_bordo=0.02):
    """Il sentiero come un NASTRO: una linea centrale che flette e una semilarghezza in prospettiva.

        centro(y)   = c0 + c1·u + c2·u²        (u = y − yref)
        semilarg(y) = h0 + h1·u

    I due bordi sono centro ∓ semilarghezza: non possono incrociarsi, e piegano insieme lungo il sentiero.
    Si adatta ai punti medi e alle semilarghezze per riga della maschera, partendo dalle rette del notebook.
    Ritorna ((c0,c1,c2), (h0,h1)) in coordinate del fotogramma, oppure None.
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
    ok = m[ys].any(axis=1) & (sx > lim) & (dx < Wf - 1 - lim) & (dx - sx > 4)      # righe con ENTRAMBI i bordi veri
    if ok.sum() < 20:
        return None
    u = ys[ok].astype(float) - yref
    pl, pr = geo
    c_nb = [(np.polyval(pl, yref) + np.polyval(pr, yref)) / 2, (pl[0] + pr[0]) / 2, 0.0]
    h_nb = [(np.polyval(pr, yref) - np.polyval(pl, yref)) / 2, (pr[0] - pl[0]) / 2]
    centro = _fit_robusto(u, (sx[ok] + dx[ok]) / 2, c_nb, 2)
    semi = _fit_robusto(u, (dx[ok] - sx[ok]) / 2, h_nb, 1)
    return tuple(float(v) for v in centro), tuple(float(v) for v in semi)


def punti_nastro(par, geo, ys, yref, quota_centro=0.5, banda_larghezza=(0.6, 1.4)):
    """Tre curve (sinistra, destra, mezzeria) alle righe `ys`, con i vincoli che le tengono sensate.

    Rispetto alle rette del notebook (validate): la semilarghezza resta fra 0,6 e 1,4 volte quella della retta,
    e il centro non si allontana dalla retta centrale più di `quota_centro` semilarghezze. Quindi le curve
    piegano, ma non possono incrociarsi né fuggire via.
    """
    (c0, c1, c2), (h0, h1) = par
    pl, pr = geo
    u = np.asarray(ys, float) - yref
    h_nb = (np.polyval(pr, ys) - np.polyval(pl, ys)) / 2
    c_nb = (np.polyval(pl, ys) + np.polyval(pr, ys)) / 2
    h = np.clip(h0 + h1 * u, banda_larghezza[0] * h_nb, banda_larghezza[1] * h_nb)
    h = np.maximum(h, 2.0)
    c = c0 + c1 * u + c2 * u * u
    c = np.clip(c, c_nb - quota_centro * h_nb, c_nb + quota_centro * h_nb)
    return c - h, c + h, c


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

    par = nastro_bordi(maschera, geo, yref, y_alto, y_basso)
    if par is None:                                   # maschera troppo povera: due rette (nastro dritto)
        par = ((float((np.polyval(pl, yref) + np.polyval(pr, yref)) / 2), float((pl[0] + pr[0]) / 2), 0.0),
               (float(largh / 2), float((pr[0] - pl[0]) / 2)))
    elif _prec["cv"] is not None:                     # liscia nel tempo: la traiettoria non salta da un fotogramma all'altro
        pa = _prec["cv"]
        par = (tuple(a + SMORZ * (b - a) for a, b in zip(pa[0], par[0])), tuple(a + SMORZ * (b - a) for a, b in zip(pa[1], par[1])))
    _prec["cv"] = par

    # campioni dal basso (vicino) verso l'alto (lontano): passo che si infittisce, come la prospettiva.
    # Si ferma dove la retta del notebook darebbe una semilarghezza quasi nulla (punto di fuga).
    n = 44
    yb = min(Hf - 1.0, y_basso + N._k(6))
    h_ref = (np.polyval(pr, yref) - np.polyval(pl, yref)) / 2
    ya = y_alto
    for y_try in np.linspace(y_alto, yb, 200):
        if (np.polyval(pr, y_try) - np.polyval(pl, y_try)) / 2 >= 0.12 * h_ref:
            ya = y_try
            break
    ys = np.array([yb + (ya - yb) * (1 - (1 - i / (n - 1)) ** 1.6) for i in range(n)])
    xl, xr, xm = punti_nastro(par, geo, ys, yref)
    L = [(int(round(a)), int(round(y))) for a, y in zip(xl, ys)]
    R = [(int(round(a)), int(round(y))) for a, y in zip(xr, ys)]
    M = [(int(round(a)), int(round(y))) for a, y in zip(xm, ys)]

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
