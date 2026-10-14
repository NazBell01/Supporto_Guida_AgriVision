"""Genera agrivision_rpi/nucleo.py dal notebook Simulatore AgriVision REV_R.

Le funzioni di geometria, il righello e il cruscotto NON sono riscritte: sono estratte dal
notebook, nodo per nodo, nell'ordine in cui il notebook le esegue (celle 2, 2bis, 2ter, 3),
così il numero mostrato sulla Raspberry è quello del notebook. Quando esce una nuova revisione:

    python3 genera_nucleo.py  Simulatore_AgriVision_REV_X.ipynb  ../agrivision_rpi/nucleo.py

Si prendono solo `def`, `class` e assegnamenti; le righe `!pip`, i `print` e tutto ciò che legge
da Drive o dal video restano fuori. Le grandezze che dipendono dall'ambiente (modello, scala del
corridoio, logo) si impostano a runtime con `nucleo.configura(...)`.
"""
import ast, json, sys

# (indice cella, nome) in ordine di esecuzione. Le definizioni successive sostituiscono le precedenti,
# come in Python: la 2bis rimpiazza _mirino, la 2ter rimpiazza _testo, _riquadro e il cruscotto.
CELLA_6 = ["MEDIA_IMGNET", "DEV_IMGNET", "preprocess_guida", "scarto_laterale", "_retta_robusta",
           "geometria_corridoio", "Righello", "C_AMBRA", "C_ROSSO", "C_GREEN_DIM", "_k",
           "C_CIANO", "C_CIANO_S", "C_ARANCIO", "C_ROSSO_S", "C_FONDO", "MARCHIO", "LOGO_MONOCROMATICO",
           "PRODOTTO", "_squadre", "_scala_cm", "_barre", "carica_logo", "_incolla",
           "LOGO_INCORPORATO_B64", "_carica_marchio", "guida"]
CELLA_8 = ["_mirino"]
CELLA_12_COLORI = ["C_GREEN", "C_DARK", "C_PANEL", "C_SLATE", "C_WHITE", "C_GRAY", "C_RED", "C_ORANGE",
                   "C_TRACK", "COLORI_AZ", "FONT", "FONT_B"]

INTESTAZIONE = '''"""Nucleo di AgriVision REV_R: geometria del corridoio, righello e cruscotto.

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
'''

CODA = '''

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
'''


def principale(notebook, destinazione):
    nb = json.load(open(notebook))
    def nodi(i):
        src = ''.join(nb['cells'][i]['source'])
        src = '\n'.join(('#' + l if l.lstrip().startswith('!') else l) for l in src.split('\n'))
        righe = src.split('\n')
        out = {}
        for n in ast.parse(src).body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                nome = n.name
            elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                nome = n.targets[0].id
            else:
                continue
            out.setdefault(nome, []).append('\n'.join(righe[n.lineno - 1:n.end_lineno]))
        return out
    c6, c8, c10, c12 = nodi(6), nodi(8), nodi(10), nodi(12)
    pezzi = [INTESTAZIONE]
    def aggiungi(tab, nome, quale=-1, etichetta=''):
        if nome not in tab:
            raise SystemExit(f"manca «{nome}» nella cella {etichetta}: il notebook è cambiato?")
        pezzi.append(tab[nome][quale])
    for n in CELLA_6:
        aggiungi(c6, n, 0, '6')
        if n == "Righello":
            pezzi.append("righello = Righello()")
    for n in CELLA_8:
        aggiungi(c8, n, -1, '8')
    # cella 10: tutto, nell'ordine di definizione, tranne gli assegnamenti a variabili di stato
    src10 = ''.join(nb['cells'][10]['source']).split('\n')
    for n in ast.parse('\n'.join(src10)).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) or (isinstance(n, ast.Assign)):
            pezzi.append('\n'.join(src10[n.lineno - 1:n.end_lineno]))
    for n in CELLA_12_COLORI:
        aggiungi(c12, n, 0, '12')
    pezzi.append(CODA)
    testo = '\n\n'.join(pezzi)
    compile(testo, destinazione, 'exec')           # deve almeno essere Python valido
    open(destinazione, 'w').write(testo)
    print(f"scritto {destinazione}: {len(testo.splitlines())} righe")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    principale(sys.argv[1], sys.argv[2])
