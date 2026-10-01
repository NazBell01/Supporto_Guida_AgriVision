# Prove

**`confronta.mjs`** — confronta la geometria JavaScript (`js/geometria.js`) con le funzioni
originali del notebook su 9 maschere sintetiche (`riferimento.json`): scarto laterale, rette dei
bordi, larghezza, righello. Non servono dipendenze: `node test/confronta.mjs`.
Gira anche in CI prima di pubblicare. `genera_riferimento.py` rigenera il file (richiede numpy;
le tre funzioni sono copiate dal notebook).

**`modello_finto.py`, `video_finto.py`, `e2e_camera.cjs`** — prova dell'intera app in Chromium
senza il modello vero: un ONNX finto (classe 1 = «verde meno rosso») e un video di un corridoio
verde che oscilla, mandato alla pagina come videocamera simulata. Verificano videocamera,
inferenza in WebAssembly, disegno e registro; **non** dicono nulla sulla qualità del modello.
Servono `onnx`, `numpy`, `pillow`, ffmpeg e Playwright; vedi i percorsi in testa ai file.

**`generale.mjs`** — prove della modalità generale: scene sintetiche (terreno libero, ostacoli a
destra/sinistra/al centro, muro vicino, parete di sbieco, telefono inclinato, camera calibrata) e tre
profondità **vere** prodotte da MiDaS su foto reali (`profondita_reali.json`): libri su un pavimento
(deve vedere gli ostacoli e indicare il libero), facciata di un palazzo e primo piano di una pianta
(non devono dare una via libera inventata). `node test/generale.mjs`.
