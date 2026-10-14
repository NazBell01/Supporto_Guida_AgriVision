"""Nucleo di AgriVision REV_R: geometria del corridoio, righello e cruscotto.

FILE GENERATO da tools/genera_nucleo.py a partire dal notebook. Non si modifica a mano:
si rigenera. Le funzioni sono quelle del notebook, parola per parola.
"""
import os, time, collections
from collections import deque
import cv2
import numpy as np

# --- Parametri della cella 1 del notebook (stessi valori) ---------------------
GUIDA_DIM = (320, 192)            # deve combaciare con IMAGE_SIZE dell'addestramento
SOGLIA_STERZO_CM = 15.0
RIGA_RIGHELLO = 0.72
SECONDI_TRACCIA = 20
EFFETTI_DECORATIVI = False
GUIDA_ERRORE_DENTRO_CM = 4.2      # misurati sulla sessione D-20
GUIDA_ERRORE_FUORI_CM = 8.2
GUIDA_ABILITA = 0.93
GUIDA_SESSIONE = "Rev D-20"
MOSTRA_CAMPI_FUTURI = False

# --- Impostati a runtime da configura() ---------------------------------------
CORRIDOIO_CM = None               # None = scala mai misurata
sess_guida = None
NOME_IN_GUIDA = None
motore_guida = "onnx"
modello_torch = None
PERCORSO_LOGO = ""
K_UI = 1.0
LOGO = None
TESTATA_H = 0
PIEDE_H = 0


MEDIA_IMGNET = np.array([0.485, 0.456, 0.406], dtype=np.float32)

DEV_IMGNET   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_guida(frame):
    """Lettera di casella + normalizzazione, come in addestramento.

    Restituisce anche il riquadro utile, che serve a rimappare la maschera sul
    fotogramma senza spalmare il riempimento sui bordi.
    """
    L, A = GUIDA_DIM
    h, w = frame.shape[:2]
    s = min(L / w, A / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    piccolo = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    tela = np.zeros((A, L, 3), dtype=np.uint8)
    ox, oy = (L - nw) // 2, (A - nh) // 2
    tela[oy:oy+nh, ox:ox+nw] = piccolo
    img = cv2.cvtColor(tela, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEDIA_IMGNET) / DEV_IMGNET
    img = np.transpose(img, (2, 0, 1))[None]
    return np.ascontiguousarray(img), (ox, oy, nw, nh)

def scarto_laterale(mask, frazione_bassa=0.35):
    """Centro orizzontale della regione percorribile nella fascia bassa, in
    pixel rispetto al centro. Positivo = corridoio a destra, cioè la macchina
    è troppo a sinistra e deve correggere verso destra.

    Identica alla funzione del notebook di addestramento: se le due divergono,
    il numero mostrato qui non è quello su cui il modello è stato valutato.
    """
    A, L = mask.shape[:2]
    fascia = mask[int(A * (1 - frazione_bassa)):, :]
    pesi = fascia.sum(axis=0).astype(np.float32)
    tot = pesi.sum()
    if tot < 1:
        return None
    colonne = np.arange(L, dtype=np.float32)
    return float((pesi * colonne).sum() / tot) - (L - 1) / 2.0

def _retta_robusta(y, x, giri=3):
    """Adatta una retta scartando i punti lontani.

    Su una maschera PREVISTA i bordi sono frastagliati e un adattamento ai
    minimi quadrati semplice seguirebbe le sbavature.
    """
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
    """Dalle due rette dei bordi ricava centro e larghezza, in pixel di tela.

    Identica alla funzione del notebook di addestramento dalla D-16 in avanti.
    Se le due divergono, i numeri mostrati qui non sono quelli su cui il modello
    è stato valutato.

    Le righe in cui un bordo tocca la cornice sono escluse dall'adattamento: lì
    il bordo non è il corridoio, è il fotogramma che finisce.
    """
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

class Righello:
    """Tiene la mediana mobile della larghezza del corridoio.

    Serve a riconoscere una stima non credibile: l'estrapolazione delle rette
    può dare un corridoio assurdamente stretto quando i bordi sono quasi
    paralleli o la maschera è molto censurata. In quei fotogrammi la conversione
    in centimetri va scartata, non usata.
    """
    def __init__(self, memoria=45):
        self.storia = collections.deque(maxlen=memoria)

    def aggiorna(self, larghezza):
        if larghezza and larghezza > 5:
            self.storia.append(float(larghezza))

    def credibile(self, larghezza):
        if not larghezza or larghezza <= 5:
            return False
        if len(self.storia) < 8:
            return True               # all'inizio non c'è un riferimento
        med = float(np.median(self.storia))
        return 0.5 * med <= larghezza <= 2.0 * med

righello = Righello()

C_AMBRA     = ( 30, 170, 240)

C_ROSSO     = ( 60,  60, 220)

C_GREEN_DIM = ( 60, 130,  25)

def _k(v):
    return max(1, int(round(v * K_UI)))

C_CIANO   = (222, 214,  46)     # BGR

C_CIANO_S = (120, 110,  24)

C_ARANCIO = ( 40, 170, 250)

C_ROSSO_S = ( 70,  70, 235)

C_FONDO   = ( 22,  16,  10)

MARCHIO   = "KYDO ROBOTICS"          # ripiego se il file del logo non c'è

LOGO_MONOCROMATICO = False   # colori originali del marchio

PRODOTTO  = "AGRIVISION  //  AUSILIO ALLA GUIDA"

def _squadre(f, col, sp=None, lung=None):
    """Le quattro squadre agli angoli. Decorative, e lo sono anche nell'originale."""
    Hf, Wf = f.shape[:2]
    sp = sp or _k(2); lung = lung or _k(46); m = _k(14)
    for (x, y, dx, dy) in ((m, m, 1, 1), (Wf-m, m, -1, 1),
                           (m, Hf-m, 1, -1), (Wf-m, Hf-m, -1, -1)):
        cv2.line(f, (x, y), (x + dx*lung, y), col, sp, cv2.LINE_AA)
        cv2.line(f, (x, y), (x, y + dy*lung), col, sp, cv2.LINE_AA)

def _scala_cm(f, y, centro_x, cm_px, col, mezzo=125):
    """La striscia graduata in alto: è la bussola dell'originale, ma misura
    centimetri veri invece che gradi che non conosciamo."""
    Hf, Wf = f.shape[:2]
    cv2.line(f, (_k(60), y), (Wf - _k(60), y), C_CIANO_S, _k(1), cv2.LINE_AA)
    for cm in range(-mezzo, mezzo + 1, 25):
        x = int(centro_x + cm / cm_px)
        if not (_k(60) < x < Wf - _k(60)):
            continue
        grande = cm % 50 == 0
        cv2.line(f, (x, y), (x, y - _k(12 if grande else 6)), col,
                 _k(2 if grande else 1), cv2.LINE_AA)
        if grande:
            e = "0" if cm == 0 else f"{abs(cm)}"
            (tw, _), _ = cv2.getTextSize(e, FONT, 0.34*K_UI, _k(1))
            cv2.putText(f, e, (x - tw//2, y - _k(18)), FONT, 0.34*K_UI,
                        C_CIANO_S, _k(1), cv2.LINE_AA)

def _barre(f, x, y, w, h, quota, col, n=8):
    """Le tacche del segnale radio dell'originale. Qui sono l'affidabilità
    geometrica della stima: quanto la larghezza del corridoio prevista somiglia
    a quella delle ultime inquadrature."""
    piene = int(round(np.clip(quota, 0, 1) * n))
    bw = max(1, (w - (n-1)*_k(3)) // n)
    for i in range(n):
        bx = x + i * (bw + _k(3))
        bh = int(h * (0.35 + 0.65 * (i + 1) / n))
        c = col if i < piene else C_CIANO_S
        cv2.rectangle(f, (bx, y + h - bh), (bx + bw, y + h), c,
                      -1 if i < piene else _k(1), cv2.LINE_AA)

def carica_logo(percorso, altezza, immagine=None):
    """Ritaglia il marchio dal fondo bianco e lo prepara per il video.

    Il file arriva su fondo bianco, che su un fotogramma di vigna sarebbe una
    toppa. Si ricava la trasparenza dal buio del pixel — quanto più scuro, tanto
    più opaco — si ritaglia al contenuto e si ridimensiona all'altezza voluta.

    Restituisce (colore, trasparenza) oppure None: se il file non c'è si ricade
    sulla scritta, invece di lasciare un buco.
    """
    if immagine is not None:
        im = immagine
    else:
        if not percorso or not os.path.exists(percorso):
            return None
        im = cv2.imread(percorso, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3 and im.shape[2] == 4:          # già con trasparenza
        rgb, a = im[:, :, :3], im[:, :, 3].astype(np.float32) / 255.0
    else:
        rgb = im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        a = (255 - cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 255.0
    ys, xs = np.where(a > 0.12)
    if ys.size == 0:
        return None
    rgb = rgb[ys.min():ys.max()+1, xs.min():xs.max()+1]
    a = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
    sc = altezza / rgb.shape[0]
    dim = (max(1, int(rgb.shape[1]*sc)), max(1, int(altezza)))
    rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)
    if LOGO_MONOCROMATICO:
        rgb = np.zeros_like(rgb) + np.array(C_CIANO, np.uint8)
    return rgb, a

def _incolla(f, logo, x, y):
    rgb, a = logo
    h, w = a.shape
    if y + h > f.shape[0] or x + w > f.shape[1]:
        return
    zona = f[y:y+h, x:x+w].astype(np.float32)
    f[y:y+h, x:x+w] = (zona * (1 - a[..., None]) +
                       rgb.astype(np.float32) * a[..., None]).astype(np.uint8)

LOGO_INCORPORATO_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8Q"
    "EBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
    "EBAQEBAQEBD/wAARCACMAacDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF"
    "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW"
    "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi"
    "4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC"
    "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm"
    "Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq"
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9U6+cf24/2h9Z+AHwttLjwbeRQeKvEF8tpp0kkSSiCJP3k8xRwVYBQsfIOGmU9q+jScV+"
    "Rf8AwUV+Jr+O/wBou88PWtxv07wVaR6TEA2Va4YCW4YDoDudYz/1xFezkOCWOxsYzV4rV/L/AIJ5eb4t4TDOUXaT0RXsf+CgP7XF"
    "/e2+nWXjaC5uruVIIIY9Dsi8sjMFVFHlckkgAepr9WfhxZ+MtP8AA2i2nxC1uPV/Eq2iNql3HCkSPct8zqioqqEUnapwCQoJ5Jr8"
    "u/8AgnL8J4/iJ8dU8W6la+bpfga3/tJtwyrXr5S2U+4PmSg9jCK/WcV2cS/VqNdYbDwUbauy79DlyJ16tJ1q8m76K7Fooor5o94K"
    "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"
)

def _carica_marchio(altezza):
    if PERCORSO_LOGO and os.path.exists(PERCORSO_LOGO):
        lg = carica_logo(PERCORSO_LOGO, altezza)
        if lg is not None:
            return lg, "file su Drive"
    try:
        import base64
        arr = np.frombuffer(base64.b64decode(LOGO_INCORPORATO_B64), np.uint8)
        lg = carica_logo(None, altezza, immagine=cv2.imdecode(arr, cv2.IMREAD_UNCHANGED))
        if lg is not None:
            return lg, "incorporato nel notebook"
    except Exception as e:
        print(f"⚠️  logo incorporato non leggibile: {type(e).__name__}: {e}")
    return None, "assente"

def guida(frame):
    """Ritorna (maschera, scarto in cm, scarto in frazione, affidabile).

    Lo scarto in CENTIMETRI è il numero che conta, e si ottiene usando il
    corridoio come righello: è largo CORRIDOIO_CM e sta nella stessa riga
    dell'immagine in cui misuriamo lo scarto. La larghezza in pixel è quella
    PREVISTA dal modello, perché sul rover la verità non esiste.

    La frazione di larghezza utile resta calcolata, ma solo per disegnare la
    barra e per continuità con la REV_K: non è una grandezza fisica.
    """
    if sess_guida is None:
        return None, None, None, False, None, None
    x, (ox, oy, nw, nh) = preprocess_guida(frame)
    if motore_guida == "torch":
        with torch.no_grad():
            logits = modello_torch(torch.from_numpy(x).cuda())["out"][0].cpu().numpy()
    else:
        logits = sess_guida.run(None, {NOME_IN_GUIDA: x})[0][0]
    m = (logits.argmax(0) == 1).astype(np.uint8)

    sc = scarto_laterale(m)                       # baricentro, fascia bassa
    centro_tela, larghezza, rette = geometria_corridoio(m)   # righello
    affidabile = righello.credibile(larghezza)
    righello.aggiorna(larghezza)

    cm = None
    if sc is not None and affidabile and CORRIDOIO_CM is not None:
        cm = sc / larghezza * CORRIDOIO_CM
    norm = None if sc is None or nw == 0 else sc / nw

    utile = m[oy:oy+nh, ox:ox+nw]
    Hf, Wf = frame.shape[:2]
    # Ingrandimento bilineare su float, sfocatura leggera e soglia, invece del
    # vicino più prossimo: da 320x192 a 1280x720 quest'ultimo produce un
    # contorno a gradini che si legge come lavoro fatto male, non come
    # risoluzione della stima.
    #
    # È un intervento SOLO sul disegno: la misura si fa sulla maschera alla
    # risoluzione del modello, qui sopra, e non su questa. Verificato che il
    # bordo si sposti di 1 px mediano e 2 al massimo su un fotogramma largo
    # 1280, con una differenza di area dello 0,12%: liscia, non abbellisce.
    # La sfocatura si fa sulla maschera PICCOLA e poi si ingrandisce: sfocare a
    # 1280x720 costava 3,4 ms per fotogramma, qui ne costa una frazione, e il
    # risultato è lo stesso perché l'ingrandimento bilineare liscia comunque.
    _p = cv2.GaussianBlur(utile.astype(np.float32), (0, 0), 0.8)
    _f = cv2.resize(_p, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
    grande = (_f >= 0.5).astype(np.uint8)

    # Geometria portata dalle coordinate della tela a quelle del fotogramma.
    # Si RIUSA la stima fatta sulla tela invece di rifarla sulla maschera
    # ingrandita: due calcoli darebbero due risultati leggermente diversi, e il
    # numero disegnato sullo schermo deve essere lo stesso su cui il modello è
    # stato valutato.
    # Le due RETTE, non due punti: così il righello si può valutare alla riga
    # che vogliamo disegnare invece che a quella di riferimento della tela. Con
    # una sorgente verticale quella mappa a y=684 su un fotogramma da 720 e
    # veniva schiacciata contro il pannello.
    geo = None
    if rette and nw > 0 and nh > 0:
        pl, pr = rette
        sx_f, sy_f = Wf / nw, Hf / nh
        # x_frame = (x_tela - ox) * sx ; y_tela = y_frame / sy + oy
        def porta(p):
            a, b = float(p[0]), float(p[1])
            return (a / sy_f * sx_f, (b + a * oy - ox) * sx_f)
        geo = (porta(pl), porta(pr))
    return grande, cm, norm, affidabile, geo, sc

def _mirino(f, cx, cy, r, col):
    """Nessun mirino: solo l'asse della macchina. Firma invariata perché il
    cruscotto la chiama; `cy`, `r` e `col` non servono più."""
    Hf = f.shape[0]
    cv2.line(f, (cx, _k(96)), (cx, Hf - _k(40)), (18, 14, 10), _k(3), cv2.LINE_AA)
    cv2.line(f, (cx, _k(96)), (cx, Hf - _k(40)), C_WHITE, _k(1), cv2.LINE_AA)

BLU_MARCHIO    = (128,  64,  11)     # #0B4080, campionato dal logo

GRIGIO_MARCHIO = ( 71,  71,  71)     # #474747, il grigio di ROBOTICS

C_CIANO   = (235, 170,  80)          # blu del marchio schiarito: tratti e testo

C_CIANO_S = (232, 230, 226)          # quasi bianco: etichette (REV_Q, era 150,146,140)

C_FONDO   = ( 34,  30,  26)

C_GREEN   = BLU_MARCHIO

C_ARANCIO = ( 40, 170, 250)          # invariati

C_ROSSO_S = ( 70,  70, 235)

FONDO_RIQUADRO = 0.86      # quanto il riquadro copre il video (era 0.65)

def _testo(f, s, org, scala, col, sp=1):
    corpo = sp + 1
    cv2.putText(f, s, org, FONT, scala, (10, 8, 6), corpo + _k(4), cv2.LINE_AA)
    cv2.putText(f, s, org, FONT, scala, col, corpo, cv2.LINE_AA)

def _riquadro(f, x, y, w, h, col, titolo=None):
    """Pannello molto scuro con bordo e titolino in alto a sinistra."""
    sub = f[max(0, y):y+h, max(0, x):x+w]
    if sub.size:
        sub[:] = (sub * (1.0 - FONDO_RIQUADRO)
                  + np.array(C_FONDO, np.float32) * FONDO_RIQUADRO).astype(np.uint8)
    cv2.rectangle(f, (x, y), (x+w, y+h), col, _k(1), cv2.LINE_AA)
    for (px, py, dx, dy) in ((x, y, 1, 1), (x+w, y, -1, 1),
                             (x, y+h, 1, -1), (x+w, y+h, -1, -1)):
        cv2.line(f, (px, py), (px + dx*_k(10), py), col, _k(2), cv2.LINE_AA)
        cv2.line(f, (px, py), (px, py + dy*_k(10)), col, _k(2), cv2.LINE_AA)
    if titolo:
        _testo(f, titolo, (x + _k(10), y + _k(16)), 0.34 * K_UI, C_CIANO_S)

ZONA_OPACITA    = 0.34    # trapezio, misura valida

ZONA_OPACITA_I  = 0.18    # trapezio, misura inaffidabile

MASCHERA_VELO   = 0.14    # maschera grezza sotto: si deve intuire, non leggere

GRIGLIA_TRASV   = 20

GRIGLIA_LONG    = 10

MOSTRA_CAMPI_FUTURI = False

CAMPI_FUTURI = {
    "VELOCITA": (None, "m/s"),
    "ASSETTO":  (None, "deg"),
    "TEMP":     (None, "degC"),
    "BATTERIA": (None, "%"),
}

def _valore_futuro(nome):
    v, u = CAMPI_FUTURI.get(nome, (None, ""))
    return ("--" if v is None else f"{v:.1f}"), u

def _trapezio(pl, pr, y_alto, y_basso, Wf):
    """I quattro vertici della zona guidabile, dalle due rette dei bordi."""
    bs_a, bd_a = float(np.polyval(pl, y_alto)),  float(np.polyval(pr, y_alto))
    bs_b, bd_b = float(np.polyval(pl, y_basso)), float(np.polyval(pr, y_basso))
    return (np.array([[bs_b, y_basso], [bs_a, y_alto],
                      [bd_a, y_alto], [bd_b, y_basso]], np.int32),
            (bs_a, bd_a, bs_b, bd_b))

def _griglia(g, pl, pr, y_alto, y_basso, ox, oy, sp, tratteggio):
    """Griglia dentro il trapezio.

    Le rette sono lineari, quindi valutarle a qualunque riga è esatto: le
    trasversali si prendono dai bordi alla riga voluta, le longitudinali
    interpolando fra i due bordi con la stessa frazione in alto e in basso.
    Nessun punto di fuga da stimare.
    """
    bs_a, bd_a = float(np.polyval(pl, y_alto)),  float(np.polyval(pr, y_alto))
    bs_b, bd_b = float(np.polyval(pl, y_basso)), float(np.polyval(pr, y_basso))
    for i in range(1, GRIGLIA_TRASV + 1):
        if tratteggio and i % 2:
            continue
        # passo che si infittisce verso l'alto: righe equidistanti
        # sembrerebbero un pavimento piatto, non un interfilare
        y = y_basso + (y_alto - y_basso) * (i / (GRIGLIA_TRASV + 1.0)) ** 1.45
        cv2.line(g, (int(np.polyval(pl, y)) - ox, int(y) - oy),
                 (int(np.polyval(pr, y)) - ox, int(y) - oy), 255, sp, cv2.LINE_AA)
    for i in range(1, GRIGLIA_LONG):
        t = i / float(GRIGLIA_LONG)
        cv2.line(g, (int(bs_b + (bd_b - bs_b) * t) - ox, int(y_basso) - oy),
                 (int(bs_a + (bd_a - bs_a) * t) - ox, int(y_alto) - oy),
                 255, sp, cv2.LINE_AA)

def _righello_corridoio(f, col, yref, bs, bd, centro_corr, cm_px):
    """Striscia graduata alla riga in cui la misura viene presa.

    Senza scala misurata niente tacche: si disegna la riga e si dichiara.
    """
    Hf, Wf = f.shape[:2]
    centro_img = (Wf - 1) / 2.0
    cv2.line(f, (max(0, int(bs)), yref), (min(Wf - 1, int(bd)), yref),
             col, _k(2), cv2.LINE_AA)
    if cm_px:
        # banda di incertezza: l'errore del modello FUORI giornata, che è il
        # numero che conta. Senza, il valore grande sembra preciso al decimo.
        xa = max(0, int(centro_img - GUIDA_ERRORE_FUORI_CM / cm_px))
        xb = min(Wf, int(centro_img + GUIDA_ERRORE_FUORI_CM / cm_px))
        ya, yb = max(0, yref - _k(30)), min(Hf, yref + _k(30))
        if xb > xa and yb > ya:
            _r = f[ya:yb, xa:xb]
            _r[:] = (_r * 0.86 + np.array(C_ARANCIO, np.float32) * 0.14).astype(np.uint8)
        for cm in range(-150, 151, 25):
            x = int(centro_corr + cm / cm_px)
            if not (bs < x < bd) or not (0 < x < Wf):
                continue
            grande = cm % 50 == 0
            cv2.line(f, (x, yref), (x, yref - _k(13 if grande else 6)), col,
                     _k(2 if grande else 1), cv2.LINE_AA)
            if grande:
                e = "0" if cm == 0 else str(abs(cm))
                (tw, _), _ = cv2.getTextSize(e, FONT, 0.34 * K_UI, _k(1))
                _testo(f, e, (x - tw // 2, yref - _k(19)), 0.34 * K_UI, C_CIANO_S)
    else:
        pass   # la console dice gia' scala non misurata

    # asse della macchina e centro del corridoio, alla stessa riga
    cv2.line(f, (int(centro_img), yref - _k(38)), (int(centro_img), yref + _k(38)),
             C_WHITE, _k(1), cv2.LINE_AA)
    cv2.drawMarker(f, (int(centro_img), yref + _k(38)), C_WHITE,
                   cv2.MARKER_TRIANGLE_UP, _k(12), _k(2), cv2.LINE_AA)
    cv2.line(f, (int(centro_corr), yref - _k(32)), (int(centro_corr), yref + _k(32)),
             col, _k(3), cv2.LINE_AA)
    cv2.circle(f, (int(centro_corr), yref), _k(5), col, -1, cv2.LINE_AA)

def _disegna_zona(f, maschera, geo, stato):
    """Trapezio pieno, maschera grezza sotto, griglia e righello.

    Restituisce gli stessi quattro valori della REV_O: il ciclo della cella 3
    non si accorge del cambio.
    """
    if maschera is None:
        return f, None, None, None
    Hf, Wf = f.shape[:2]
    col = {"ok": C_CIANO, "incerto": C_ARANCIO, "assente": C_ROSSO_S}[stato]
    incerto = (stato != "ok")

    m8 = (maschera > 0).astype(np.uint8)
    bx, by, bw, bh = cv2.boundingRect(m8)
    if bw < 8 or bh < 8:
        return f, None, None, None

    # --- maschera grezza: velo appena percettibile -------------------------
    roi, mroi = f[by:by+bh, bx:bx+bw], m8[by:by+bh, bx:bx+bw]
    tinta = np.empty_like(roi)
    tinta[:] = BLU_MARCHIO
    cv2.copyTo(cv2.addWeighted(roi, 1.0 - MASCHERA_VELO, tinta, MASCHERA_VELO, 0.0),
               mroi, roi)

    if geo is None:
        return f, None, None, None
    pl, pr = geo
    righe = np.flatnonzero(m8.any(1))
    y_alto, y_basso = float(righe.min()), float(righe.max())

    # --- trapezio pieno ----------------------------------------------------
    poli, _ = _trapezio(pl, pr, y_alto, y_basso, Wf)
    tz = np.zeros((Hf, Wf), np.uint8)
    cv2.fillPoly(tz, [poli], 255)
    tx, ty, tw, th = cv2.boundingRect(tz)
    if tw < 8 or th < 8:
        return f, None, None, None
    roi = f[ty:ty+th, tx:tx+tw]
    troi = tz[ty:ty+th, tx:tx+tw]
    tinta = np.empty_like(roi)
    tinta[:] = BLU_MARCHIO
    a = ZONA_OPACITA_I if incerto else ZONA_OPACITA
    cv2.copyTo(cv2.addWeighted(roi, 1.0 - a, tinta, a, 0.0), troi, roi)

    # --- griglia, ritagliata nel trapezio ----------------------------------
    g = np.zeros((th, tw), np.uint8)
    _griglia(g, pl, pr, y_alto, y_basso, tx, ty, _k(1), incerto)
    cv2.bitwise_and(g, troi, dst=g)
    tinta[:] = col
    cv2.copyTo(cv2.addWeighted(roi, 0.45, tinta, 0.55, 0.0), g, roi)

    # --- riga di misura ----------------------------------------------------
    yref = int(np.clip(RIGA_RIGHELLO * Hf, TESTATA_H + _k(50), Hf - PIEDE_H - _k(40)))
    bs, bd = float(np.polyval(pl, yref)), float(np.polyval(pr, yref))
    largh = bd - bs
    if largh < 20:
        return f, None, None, None
    centro_corr = (bs + bd) / 2.0
    cm_px = (CORRIDOIO_CM / largh) if CORRIDOIO_CM else None
    _righello_corridoio(f, col, yref, bs, bd, centro_corr, cm_px)
    return f, cm_px, centro_corr, largh

TESTATA_H = 0      # riempiti alla prima chiamata, servono a _disegna_zona

PIEDE_H   = 0

def _quadrante(f, cx, cy, r, valore, fondo, col, soglia=None):
    """Quadrante ad arco per lo scarto.

    Le tacche fuori dalla soglia di sterzo sono ambra: si vede a colpo d'occhio
    quando il numero esce dalla banda in cui lo sterzo inseguirebbe il rumore.
    Senza scala misurata la soglia non esiste, perché è in centimetri e qui i
    numeri sono pixel: in quel caso le tacche restano tutte dello stesso colore
    invece di fingere un riferimento.
    """
    a0, a1 = 200, 340
    lim = (soglia / float(fondo)) if soglia else None
    for i in range(33):
        t = i / 32.0
        ang = np.radians(a0 + (a1 - a0) * t)
        lungo = (i % 8 == 0)
        l = r * (0.24 if lungo else 0.12)
        c = col if (lim is None or abs(t - 0.5) * 2 <= lim) else C_ARANCIO
        cv2.line(f, (int(cx + np.cos(ang) * r), int(cy + np.sin(ang) * r)),
                 (int(cx + np.cos(ang) * (r - l)), int(cy + np.sin(ang) * (r - l))),
                 c, _k(2 if lungo else 1), cv2.LINE_AA)
    for t, e in ((0.0, f"-{fondo:.0f}"), (1.0, f"+{fondo:.0f}")):
        ang = np.radians(a0 + (a1 - a0) * t)
        (tw, _), _ = cv2.getTextSize(e, FONT, 0.26 * K_UI, _k(1))
        _testo(f, e, (int(cx + np.cos(ang) * (r + _k(4))) - (tw if t == 0 else 0),
                      int(cy + np.sin(ang) * (r + _k(4))) + _k(10)),
               0.26 * K_UI, C_CIANO_S)
    if valore is None:
        return
    t = float(np.clip(valore / fondo, -1, 1)) * 0.5 + 0.5
    ang = np.radians(a0 + (a1 - a0) * t)
    cv2.line(f, (cx, cy), (int(cx + np.cos(ang) * r * 0.9),
                           int(cy + np.sin(ang) * r * 0.9)),
             C_WHITE, _k(2), cv2.LINE_AA)
    cv2.circle(f, (cx, cy), _k(3), C_WHITE, -1, cv2.LINE_AA)

def _traccia(f, x, y, w, h, storia, col, stati=None, scala=100.0, soglia=None):
    """La traccia dello scarto lungo gli ultimi secondi.

    Non c'è nell'immagine di riferimento ed è l'elemento più informativo del
    cruscotto: è l'unico che mostra se la stima trema invece di misurare.

    Tre scelte, tutte per non far dire alla traccia più di quello che sa.

    L'asse del tempo è FISSO: il fotogramma più recente sta a destra e la traccia
    si riempie da destra. Prima si stirava su tutta la larghezza a ogni
    fotogramma, quindi nei primi venti secondi la stessa oscillazione sembrava
    più lenta man mano che la storia cresceva.

    La banda chiara è la zona in cui lo sterzo resta DRITTO (la soglia in
    centimetri): senza, la traccia mostra un numero che si muove ma non dice se
    quel movimento cambierebbe una decisione. Senza scala misurata la soglia non
    esiste e la banda non si disegna.

    Le tacche sul fondo segnano i fotogrammi senza misura valida, ambra per
    «inaffidabile» e rossa per «nessun corridoio»: un buco nella linea da solo
    non dice PERCHÉ manca il dato.
    """
    my = y + h // 2
    if soglia:
        mezza = int(h / 2 * min(1.0, soglia / float(scala)))
        _r = f[max(0, my - mezza):my + mezza + 1, x:x + w]
        if _r.size:
            _r[:] = (_r * 0.80 + np.array(C_CIANO_S, np.float32) * 0.20).astype(np.uint8)
    cv2.line(f, (x, my), (x + w, my), C_CIANO_S, _k(1), cv2.LINE_AA)
    n = len(storia)
    n_max = getattr(storia, "maxlen", None) or n
    if n < 2 or n_max < 2:
        return
    passo = w / float(n_max - 1)
    xs = x + w - (n - 1 - np.arange(n)) * passo
    a = np.array([v if v is not None else np.nan for v in storia], float)
    ys = my - np.clip(a, -scala, scala) / float(scala) * (h // 2)
    for i in range(1, n):
        if np.isnan(ys[i-1]) or np.isnan(ys[i]):
            continue
        cv2.line(f, (int(xs[i-1]), int(ys[i-1])), (int(xs[i]), int(ys[i])),
                 col, _k(1), cv2.LINE_AA)
    if stati is not None:
        base = y + h
        for i, s in enumerate(list(stati)[:n]):
            if s == "ok":
                continue
            cv2.line(f, (int(xs[i]), base), (int(xs[i]), base - _k(4)),
                     C_ARANCIO if s == "incerto" else C_ROSSO_S, _k(1))
    if not np.isnan(ys[-1]):
        cv2.circle(f, (int(xs[-1]), int(ys[-1])), _k(2), col, -1, cv2.LINE_AA)

def _geom_console(Hf, Wf):
    """Dove sta la console unica in basso. Il righello deve restare sopra."""
    m, g = _k(26), _k(8)
    Wc = Wf - 2 * m
    ph = _k(90)                       # altezza di una fila di contenuti
    if Wf < Hf:                       # verticale: due file (scarto | corridoio) e sotto la traccia
        wp, ph_t = (Wc - g) // 2, _k(54)
        ph_c = ph + ph_t
        yc0 = Hf - _k(34) - ph_c
        return dict(m=m, Wc=Wc, ph=ph, ph_c=ph_c, yc0=yc0, vert=True,
                    s=(m, yc0, wp), r=(m + wp + g, yc0, Wc - wp - g),
                    t=(m, yc0 + ph + g, Wc, ph_t))
    wp = int(Wc * 0.30)               # orizzontale: tre colonne
    yc0 = Hf - _k(34) - ph
    return dict(m=m, Wc=Wc, ph=ph, ph_c=ph, yc0=yc0, vert=False,
                s=(m, yc0, wp), r=(m + wp + g, yc0, int(Wc * 0.26)),
                t=(m + wp + g + int(Wc * 0.26) + g, yc0, Wc - wp - int(Wc * 0.26) - 2 * g, ph))

def disegna_cruscotto_strumentale(f, dati):
    """Console unica in basso, video pieno sopra, senza mirino.

    Ogni riquadro è agganciato a un numero che il modello calcola davvero. I
    campi che arriveranno con i sensori restano vuoti e dichiarati tali.
    """
    global TESTATA_H, PIEDE_H
    Hf, Wf = f.shape[:2]
    stato = dati["stato"]
    col = {"ok": C_CIANO, "incerto": C_ARANCIO, "assente": C_ROSSO_S}[stato]
    m = _k(26)
    TESTATA_H = _k(74)
    PIEDE_H = Hf - _geom_console(Hf, Wf)["yc0"]
    _squadre(f, col)

    # --- testata ------------------------------------------------------------
    if LOGO is not None:
        # Piastra CHIARA sotto il marchio: il logo è disegnato per fondo bianco
        # e su una piastra scura il blu si spegne. È il resto del cruscotto ad
        # adattarsi al marchio, non il contrario.
        _lh, _lw = LOGO[1].shape
        _px, _py, _mm = m + _k(4), _k(16), _k(10)
        _z = f[max(0, _py-_mm):_py+_lh+_mm, max(0, _px-_mm):_px+_lw+_mm]
        if _z.size:
            _z[:] = (_z * 0.10 + np.float32(248) * 0.90).astype(np.uint8)
            cv2.rectangle(f, (_px-_mm, _py-_mm), (_px+_lw+_mm, _py+_lh+_mm),
                          col, _k(1), cv2.LINE_AA)
        _incolla(f, LOGO, _px, _py)
        xt = _px + _lw + _k(22)
    else:
        # La larghezza si MISURA: stimarla a occhio faceva finire «AGRIVISION»
        # sopra la scritta del marchio ogni volta che il file del logo mancava.
        (mw, _), _ = cv2.getTextSize(MARCHIO, FONT, 0.58 * K_UI, _k(2))
        _testo(f, MARCHIO, (m + _k(4), _k(38)), 0.58 * K_UI, col, _k(2))
        xt = m + _k(4) + mw + _k(22)
    # Nessun'altra scritta in intestazione: solo il marchio. Acquisizione, vigna,
    # versione e FPS restano nel nome del file e nel json della simulazione.

    # --- solo l'asse della macchina (il mirino non esiste più) --------------
    _mirino(f, Wf // 2, int(Hf * 0.46), _k(50), col)

    # --- console unica in basso ----------------------------------------------
    # I due riquadri in alto coprivano il video: stesso contenuto, ridotto
    # all'indispensabile e riunito in un solo blocco. Tolti: errore modello, base
    # in cm, barre di affidabilita (lo stato resta nella riga in fondo).
    G = _geom_console(Hf, Wf)
    xs, ys, ws = G["s"]
    xr, yr, wr = G["r"]
    xtr, ytr, wtr, htr = G["t"]
    _riquadro(f, m, G["yc0"], G["Wc"], G["ph_c"], col, None)
    for xsep in ((xs + ws + _k(4), xr + wr + _k(4)) if not G["vert"] else (xs + ws + _k(4),)):
        if xsep < m + G["Wc"] - _k(20):
            cv2.line(f, (xsep, G["yc0"] + _k(8)), (xsep, G["yc0"] + G["ph"] - _k(8)),
                     C_CIANO_S, _k(1), cv2.LINE_AA)
    if G["vert"]:
        cv2.line(f, (m + _k(8), ytr - _k(4)), (m + G["Wc"] - _k(8), ytr - _k(4)),
                 C_CIANO_S, _k(1), cv2.LINE_AA)

    # scarto laterale: numero, da che parte sta il corridoio, barra con zona DRITTO
    _testo(f, "SCARTO LATERALE", (xs + _k(10), ys + _k(16)), 0.34 * K_UI, C_CIANO_S)
    v = dati["scarto_cm"]
    a = np.array([x for x in dati["storia"] if x is not None], float)
    if dati["unita"] == "cm":
        fondo, soglia = max(3 * SOGLIA_STERZO_CM, 30.0), SOGLIA_STERZO_CM
    else:
        fondo = float(max(30.0, np.percentile(np.abs(a), 95) * 1.3)) if a.size else 30.0
        soglia = None
    testo = f"{abs(v):.0f}" if v is not None else "--"
    _testo(f, testo, (xs + _k(12), ys + _k(46)), 0.92 * K_UI, col, _k(2))
    (nw_, _), _ = cv2.getTextSize(testo, FONT, 0.92 * K_UI, _k(2))
    _testo(f, dati["unita"], (xs + _k(18) + nw_, ys + _k(46)), 0.40 * K_UI, C_CIANO_S)
    if v is None:
        frase = {"incerto": "MISURA INAFFIDABILE", "assente": "NESSUN CORRIDOIO"}.get(stato, "--")
        col_f = C_ARANCIO if stato == "incerto" else C_ROSSO_S
    elif soglia is not None and abs(v) <= soglia:
        frase, col_f = "CORRIDOIO CENTRATO", col
    else:
        frase, col_f = ("CORRIDOIO A DESTRA" if v > 0 else "CORRIDOIO A SINISTRA"), col
    _testo(f, frase, (xs + _k(12), ys + _k(64)), 0.34 * K_UI, col_f)
    bx0, bx1 = xs + _k(14), xs + ws - _k(10)
    by0, by1 = ys + _k(72), ys + _k(82)
    cxb, semi = (bx0 + bx1) // 2, (bx1 - bx0) / 2.0
    if soglia is not None:
        za, zb = int(cxb - semi * soglia / fondo), int(cxb + semi * soglia / fondo)
        _z = f[by0:by1, za:zb]
        if _z.size:
            _z[:] = (_z * 0.55 + np.array(C_CIANO_S, np.float32) * 0.45).astype(np.uint8)
    if v is not None:
        mx = int(cxb + np.clip(v / fondo, -1.0, 1.0) * semi)
        cv2.rectangle(f, (min(cxb, mx), by0), (max(cxb, mx), by1),
                      C_ARANCIO if abs(v) > fondo else col, -1)
    cv2.rectangle(f, (bx0, by0), (bx1, by1), C_CIANO_S, _k(1), cv2.LINE_AA)
    cv2.line(f, (cxb, by0 - _k(3)), (cxb, by1 + _k(3)), C_CIANO_S, _k(1), cv2.LINE_AA)
    if v is not None:
        cv2.line(f, (mx, by0 - _k(4)), (mx, by1 + _k(4)), (10, 8, 6), _k(5), cv2.LINE_AA)
        cv2.line(f, (mx, by0 - _k(4)), (mx, by1 + _k(4)), C_WHITE, _k(3), cv2.LINE_AA)

    # corridoio: la sola stima di larghezza
    _testo(f, "CORRIDOIO", (xr + _k(10), yr + _k(16)), 0.34 * K_UI, C_CIANO_S)
    _st = dati.get("stima_cm")
    t2 = f"~{_st:.0f}" if (CORRIDOIO_CM and _st is not None) else "--"
    _testo(f, t2, (xr + _k(12), yr + _k(46)), 0.92 * K_UI, C_CIANO, _k(2))
    (n2, _), _ = cv2.getTextSize(t2, FONT, 0.92 * K_UI, _k(2))
    _testo(f, "cm  stima" if CORRIDOIO_CM else "px",
           (xr + _k(18) + n2, yr + _k(46)), 0.34 * K_UI, C_CIANO_S)
    if not CORRIDOIO_CM:
        _testo(f, "scala non misurata", (xr + _k(12), yr + _k(64)), 0.28 * K_UI, C_ARANCIO)

    # traccia dello scarto
    _testo(f, f"SCARTO  {SECONDI_TRACCIA}s", (xtr + _k(10), ytr + _k(16)),
           0.34 * K_UI, C_CIANO_S)
    _traccia(f, xtr + _k(10), ytr + _k(22), wtr - _k(20), htr - _k(30) if G["vert"] else G["ph"] - _k(30),
             dati["storia"], col, dati.get("storia_stati"),
             scala=100.0 if dati["unita"] == "cm" else max(30.0, fondo),
             soglia=soglia)

    # --- riga di stato ------------------------------------------------------
    et = {"ok": "MISURA VALIDA", "incerto": "MISURA INAFFIDABILE",
          "assente": "NESSUN CORRIDOIO"}[stato]
    t3 = f"AUSILIO ALL'OPERATORE  //  {et}"
    (ew, _), _ = cv2.getTextSize(t3, FONT, 0.38 * K_UI, _k(1))
    _testo(f, t3, ((Wf - ew) // 2, Hf - _k(16)), 0.38 * K_UI, col)
    cv2.circle(f, (m + _k(6), Hf - _k(20)), _k(5), C_ARANCIO, -1, cv2.LINE_AA)
    _testo(f, "REC", (m + _k(18), Hf - _k(16)), 0.30 * K_UI, C_CIANO_S)
    return f

GRIGIO_CHIARO = (170, 168, 164)      # i trattini dei campi non misurati

disegna_corridoio = _disegna_zona

_nota_mira = "senza mirino (resta solo la riga dell'asse della macchina)"

C_GREEN  = (94,  197, 34)

C_DARK   = (32,  18,  11)

C_PANEL  = (48,  30,  20)

C_SLATE  = (59,  41,  30)

C_WHITE  = (249, 245, 241)

C_GRAY   = (139, 116, 100)

C_RED    = (68,  68,  239)

C_ORANGE = (30,  140, 255)

C_TRACK  = (55,  42,  33)

COLORI_AZ = {"DRITTO": C_GREEN, "FRENA": C_RED, "DESTRA": C_ORANGE, "SINISTRA": C_ORANGE}

FONT   = cv2.FONT_HERSHEY_SIMPLEX

FONT_B = cv2.FONT_HERSHEY_DUPLEX



# ===========================================================================
# Impostazione a runtime (unica parte non presente nel notebook)
# ===========================================================================
def configura(sessione_onnx, corridoio_cm, larghezza_fotogramma, percorso_logo=""):
    """Collega il modello e la scala della vigna, e fissa la scala del cruscotto
    sulla larghezza del fotogramma, come fa la cella 3 del notebook."""
    global sess_guida, NOME_IN_GUIDA, CORRIDOIO_CM, K_UI, LOGO, PERCORSO_LOGO, righello
    sess_guida = sessione_onnx
    NOME_IN_GUIDA = sessione_onnx.get_inputs()[0].name
    CORRIDOIO_CM = corridoio_cm
    PERCORSO_LOGO = percorso_logo or ""
    K_UI = float(np.clip((larghezza_fotogramma / 404.0) ** 0.6, 0.9, 2.2))
    LOGO, _origine = _carica_marchio(_k(30))
    righello = Righello()
    return _origine
