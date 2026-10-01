"""Genera test/riferimento.json: maschere sintetiche e risultati attesi.

Le tre funzioni sotto sono copiate VERBATIM dal notebook REV_R (celle 2 e 3),
così il porting JavaScript si confronta con il codice originale.
Uso:  python genera_riferimento.py   (richiede solo numpy)
"""
import json, base64, collections
import numpy as np

# ---- dal notebook --------------------------------------------------------
def scarto_laterale(mask, frazione_bassa=0.35):
    A, L = mask.shape[:2]
    fascia = mask[int(A * (1 - frazione_bassa)):, :]
    pesi = fascia.sum(axis=0).astype(np.float32)
    tot = pesi.sum()
    if tot < 1:
        return None
    colonne = np.arange(L, dtype=np.float32)
    return float((pesi * colonne).sum() / tot) - (L - 1) / 2.0

def _retta_robusta(y, x, giri=3):
    p = np.polyfit(y, x, 1)
    for _ in range(giri):
        r = np.abs(x - np.polyval(p, y))
        s = max(float(np.median(r)), 0.5)
        k = r < 4 * s
        if k.sum() < 8:
            break
        p = np.polyfit(y[k], x[k], 1)
    return p

def geometria_corridoio(m, margine_alto=0.05, margine_bordo=0.02):
    mm = np.asarray(m, dtype=bool)
    A, L = mm.shape[-2:]
    presenti = np.where(mm.any(1))[0]
    if presenti.size < 20:
        return None, None, None
    ap = int(presenti.min())
    sx = np.argmax(mm, axis=1)
    dx = L - 1 - np.argmax(mm[:, ::-1], axis=1)
    lim = max(1.0, margine_bordo * L)
    uso = [y for y in presenti if ap + margine_alto * A <= y <= A - 0.03 * A]
    ls = np.array([y for y in uso if sx[y] > lim], dtype=float)
    ld = np.array([y for y in uso if dx[y] < L - lim], dtype=float)
    if ls.size < 10 or ld.size < 10:
        return None, None, None
    pl = _retta_robusta(ls, sx[ls.astype(int)].astype(float))
    pr = _retta_robusta(ld, dx[ld.astype(int)].astype(float))
    yref = 0.95 * A
    bs, bd = float(np.polyval(pl, yref)), float(np.polyval(pr, yref))
    return (bs + bd) / 2.0 - (L - 1) / 2.0, bd - bs, (pl, pr)
# --------------------------------------------------------------------------

L, A = 320, 192
rng = np.random.default_rng(7)

def corridoio(top_l, top_r, bot_l, bot_r, y0=40, rumore=0.0, buchi=0):
    m = np.zeros((A, L), np.uint8)
    for y in range(y0, A):
        t = (y - y0) / (A - 1 - y0)
        a = top_l + (bot_l - top_l) * t
        b = top_r + (bot_r - top_r) * t
        if rumore:
            a += rng.normal(0, rumore); b += rng.normal(0, rumore)
        a, b = int(round(max(0, a))), int(round(min(L - 1, b)))
        if b > a:
            m[y, a:b + 1] = 1
    for _ in range(buchi):
        x, y = rng.integers(40, 280), rng.integers(60, 180)
        m[y:y + 6, x:x + 8] = 1 - m[y:y + 6, x:x + 8]
    return m

casi = [
    ("centrato",        corridoio(140, 180,  70, 250)),
    ("a_destra",        corridoio(170, 210, 110, 300)),
    ("a_sinistra",      corridoio( 90, 130,  20, 210)),
    ("rumoroso",        corridoio(140, 180,  70, 250, rumore=2.0, buchi=15)),
    ("stretto",         corridoio(150, 160, 120, 200)),
    ("tocca_bordo",     corridoio(100, 220,   0, 319)),
    ("quasi_vuoto",     corridoio(150, 170, 140, 180, y0=170)),
    ("vuoto",           np.zeros((A, L), np.uint8)),
    ("con_macchie",     corridoio(130, 190,  60, 260, rumore=3.0, buchi=40)),
]
out = []
for nome, m in casi:
    sc = scarto_laterale(m)
    c, l, rette = geometria_corridoio(m)
    out.append(dict(
        nome=nome, L=L, A=A,
        bits=base64.b64encode(np.packbits(m.reshape(-1)).tobytes()).decode(),
        scarto=sc, centro=c, larghezza=l,
        rette=None if rette is None else [list(map(float, rette[0])), list(map(float, rette[1]))]))
json.dump(out, open("riferimento.json", "w"))
print(len(out), "casi scritti")
