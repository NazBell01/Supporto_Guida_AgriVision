# AgriVision · simulatore in tempo reale (telefono)

Porting nel browser del notebook **Simulatore AgriVision REV_R**. La videocamera del telefono
viene analizzata *sul telefono*, senza inviare nulla in rete: lettera di casella 320×192 →
DeepLabV3 (ONNX, onnxruntime-web: WebGPU se c'è, altrimenti WebAssembly) → scarto laterale e
geometria del corridoio → cruscotto Kydo (trapezio, griglia, righello in cm, console con scarto,
stima del corridoio e traccia dei 20 s).

## Messa online (GitHub Pages)

1. Metti il modello in `modelli/guida.onnx` (vedi `modelli/LEGGIMI.md`).
2. Su GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Fai push su `main`. L'indirizzo sarà `https://<utente>.github.io/Supporto_Guida_AgriVision/`.
4. Sul telefono apri l'indirizzo (serve **https**: la videocamera non si apre altrimenti),
   tocca *Avvia videocamera* e consenti l'accesso. Poi *Aggiungi a Home* per usarlo a schermo
   intero e **senza rete**: dopo il primo caricamento pagina, runtime e modello restano in cache.

In locale: `npx http-server . -p 8080` e `http://localhost:8080` (localhost vale come https).

## Cosa è identico al notebook e cosa no

Identici (e verificati contro il codice Python, vedi `test/`): preelaborazione, `scarto_laterale`,
`geometria_corridoio`, `_retta_robusta`, `Righello`, stati *valida / inaffidabile / nessun corridoio*,
conversione in centimetri, stima della larghezza, banda di incertezza a 8,2 cm.

Diversi:
- **Ridimensionamento.** Il notebook usa `cv2.resize`; il browser usa il ridimensionamento del
  canvas. Piccole differenze di pixel nell'ingresso possono spostare la maschera di poco: **prima di
  fidarsi dei centimetri va confrontato lo stesso filmato nel notebook e nell'app** (usa *Prova con
  un video*).
- **Orientamento.** Il modello vede 320×192, cioè un fotogramma orizzontale. Col telefono in
  verticale l'area utile si riduce a una colonna stretta e la stima peggiora: tieni il telefono
  in orizzontale (l'app lo segnala).
- **Scala.** I 200 cm sono il metro di Vigna94. Per un'altra vigna cambia *Larghezza del corridoio*
  nelle impostazioni; vuoto = lo scarto è in pixel e il cruscotto dichiara «scala non misurata».
  Con la camera del telefono a un'altezza e inclinazione diverse da quelle dell'addestramento il
  numero in centimetri **non è validato**.
- **Traccia.** Asse del tempo reale (non per fotogramma), perché in diretta i fotogrammi elaborati
  al secondo cambiano da telefono a telefono.
- **Registro.** Il pulsante ⤓ scarica `agrivision_<data>__scarti.json` con lo stesso formato del
  notebook (`scarti` e un record per fotogramma).

Come nel notebook: **ausilio all'operatore, non un comando alla macchina**. Il modello sbaglia 4,2 cm
su corridoi mai visti della stessa giornata e 8,2 cm su una giornata diversa.
