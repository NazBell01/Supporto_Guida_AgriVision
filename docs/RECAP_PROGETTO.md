# Recap del progetto · AgriVision, supporto alla guida

Periodo: 1–14 ottobre 2026. Questo documento riassume tutto quello che è stato fatto nella sessione di lavoro:
cosa esiste, dove sta, come si usa, cosa è stato verificato, cosa **non** lo è, e cosa resta da fare.

---

## 1. Punto di partenza

- Il repository `Simulatore_Macfrut` contiene la **simulazione multi-agente per la fiera Macfrut** (`simulatore.py`,
  `agenti_macfrut.json`, scheda prodotti Kydo Robotics). È un progetto diverso e **non è stato toccato** su `main`.
- Il materiale caricato era il notebook Colab **«Simulatore AgriVision REV_R»**: una pipeline in Python che
  1. legge un filmato di vigna,
  2. segmenta il corridoio percorribile con un modello **DeepLabV3 · MobileNetV3-Large** (2 classi, ingresso 320×192,
     normalizzazione ImageNet, esportato in ONNX dalla cella 1bis),
  3. ricava **scarto laterale** e **larghezza del corridoio** da due rette adattate ai bordi della maschera,
  4. converte lo scarto in **centimetri** usando la larghezza reale del corridoio (200 cm per Vigna94),
  5. disegna il **cruscotto Kydo** (marchio, righello, console con scarto, stima del corridoio e traccia dei 20 s).
- Numeri dichiarati dal notebook per la sessione **D-20** (06/09/2026): errore **4,2 cm** su corridoi mai visti della
  stessa giornata di addestramento, **8,2 cm** su una giornata diversa; soglia di sterzo DRITTO ±15 cm; abilità 0,93.
  È dichiarato come **ausilio all'operatore, non un comando alla macchina**.

## 2. Richiesta e traguardi raggiunti

| # | Richiesta | Risultato |
|---|---|---|
| 1 | Portare il simulatore su GitHub per il rilevamento **in tempo reale con la videocamera del cellulare** | App web statica su GitHub Pages, tutto in locale sul telefono |
| 2 | Repository **separato** da `Simulatore_Macfrut` | `Supporto_Guida_AgriVision` (pubblico) |
| 3 | Generalizzare: **ostacoli e percorso ottimale**, non solo vigna | «Modalità generale (prova)» con profondità monoculare |
| 4 | **Scaricare il simulatore e caricarlo su Raspberry** | Programma Python per Raspberry Pi (`raspberry/`) |
| 5 | Visione come **line assistant da retrocamera** (curve laterali + mezzeria di colore diverso) | Stile «linee guida» a nastro, in entrambe le versioni |
| 6 | Esempio visivo per capire | Immagine di confronto fra stile ad area e stile a linee |
| 7 | Prove con **video reali** (due vigne) | Diagnosi e correzioni su avvio, HEVC, incroci e spigoli |

## 3. Dove sta tutto

### Repository e link
- **Repository del progetto:** <https://github.com/NazBell01/Supporto_Guida_AgriVision> (pubblico, branch `main`)
- **App online (GitHub Pages):** <https://nazbell01.github.io/Supporto_Guida_AgriVision/>
- **Repository della fiera:** <https://github.com/NazBell01/Simulatore_Macfrut> — la prima versione web era stata
  messa sul branch `claude/friendly-hypatia-kzwgyy`, poi **rimossa** da lì e spostata nel repository nuovo; il suo `main`
  è intatto.
- **Modello di guida su Drive** (revisione D-20, 44 MB, esportato il 05/09/2026):
  <https://drive.google.com/file/d/1z3uOogEpjF9vAgL_rWdj3_uADQiQ6WYu/view> — cartella della run:
  <https://drive.google.com/drive/folders/1XQsQX6J6svEnYoyPK4CknaRloq6aq7gb>. Nella stessa cartella c'è anche il
  `.pth` (per PyTorch: **non** serve qui) e la scheda dell'esportazione con le impronte e la verifica di fedeltà
  (differenza sui logit ≈ 2·10⁻⁶, accordo sui pixel 1,0).
- **Notebook sorgente:** `Simulatore_AgriVision_REV_R.ipynb` su Drive (Colab).

### Struttura del repository `Supporto_Guida_AgriVision`
```
index.html, style.css, sw.js, manifest.webmanifest, icona-192/512.png   app web (PWA, funziona offline)
js/
  app.js          ciclo principale: sorgenti video, modelli, modalità, registro, diagnostica
  geometria.js    geometria del corridoio, righello, nastro delle linee guida (porting del notebook)
  disegno.js      cruscotto su canvas (trapezio/linee, righello, console, traccia, riga di stato)
  generale.js     modalità generale: piano di terreno, ostacoli, spazio libero, direzione
  config.js       parametri (modello, scala, errori D-20, ecc.)
vendor/ort/       onnxruntime-web 1.22.0 (WebGPU + WebAssembly), incluso nel repo
modelli/          profondita.onnx (MiDaS small, MIT), LEGGIMI.md, MiDaS-LICENSE.txt   (guida.onnx NON incluso)
test/             confronta.mjs, generale.mjs, curve.mjs, maschere.mjs + fixture e generatori di video
raspberry/        versione Python per Raspberry Pi (vedi §6)
.github/workflows/pages.yml   prove + pubblicazione su Pages (a ogni push su main)
```

### Branch creati durante il lavoro
Tutti già **fusi in `main`**: `modalita-generale`, `raspberry`, `linee-guida`, `diagnosi-avvio`, `video-non-leggibile`,
`nastro-guida`, `linee-lisce`. Ultimo commit su `main`: `5ee238a`.

---

## 4. App web per il telefono (modalità Vigna)

**Come funziona:** `getUserMedia` (camera posteriore, 1280×720) o un file video → lettera di casella 320×192 +
normalizzazione ImageNet → DeepLabV3 in **onnxruntime-web** (WebGPU se la GPU risponde davvero, altrimenti
WebAssembly) → **geometria del notebook** → cruscotto su canvas. **Nessun fotogramma lascia il dispositivo.**

- **Fedeltà al notebook:** `scartoLaterale`, `geometriaCorridoio`, `_retta_robusta` e `Righello` sono riscritte in
  JavaScript e confrontate con le funzioni **originali Python** su 9 maschere sintetiche (+ test del righello): tutto
  coincide (`test/confronta.mjs`).
- **Cruscotto:** palette del marchio (blu `#0B4080` / azzurro per linee e testo), console unica in basso (orizzontale su
  tre colonne, verticale su due file), traccia dei 20 s con asse del tempo **reale** (in diretta i fps variano),
  banda DRITTO, riga di stato «Ausilio all'operatore // misura valida / inaffidabile / nessun corridoio», banda di
  incertezza a 8,2 cm sul righello.
- **Impostazioni:** larghezza del corridoio in cm (vuoto = scala non misurata, lo scarto resta in pixel e il cruscotto
  lo dichiara), soglia di sterzo, **stile della guida** (linee / area).
- **Registro:** il pulsante ⤓ scarica `agrivision_<data>__scarti.json` (chiave `scarti` + un record per fotogramma),
  stesso formato del notebook.
- **Offline:** service worker + manifest: dopo il primo caricamento pagina, runtime e modello restano in cache e l'app
  si installa sulla schermata Home (utile dove non c'è campo).
- **Modello:** `guida.onnx` **non** è nel repository (pubblico = scaricabile da chiunque). Si sceglie con un selettore
  di file a ogni apertura; `.gitignore` impedisce di caricarlo per sbaglio.
- **Comandi in uso:** pausa, schermo intero, scarica registro, cambio camera, chiudi.

### Come si usa (telefono o computer)
1. Apri la pagina (https obbligatorio per la videocamera). Se vedi una versione vecchia: chiudi e riapri la scheda, o
   svuota i dati del sito.
2. Scegli **Vigna**, tocca «Scegli il file guida.onnx» e seleziona `drivable_migliore.onnx` (scaricato da Drive).
3. «Avvia videocamera» oppure «Prova con un video». Tieni il telefono **in orizzontale**.
4. Imposta la larghezza del corridoio se non è Vigna94.

---

## 5. Modalità generale (prova): ostacoli e direzione consigliata

Richiesta: non essere vincolati alla vigna, rilevare ostacoli e cercare il percorso migliore.

**Modello:** **MiDaS small v2.1** (licenza MIT, 64 MB, ingresso 256×256) in `modelli/profondita.onnx`. Depth Anything V2
è stato scartato: 518×518 fisso e 99 MB, troppo pesante per un telefono in tempo reale.

**Algoritmo (nessun addestramento) — v-disparità:**
1. su un terreno piano la disparità cresce linearmente verso il basso; si adatta un **piano** `d = a·v + c·x + b` alla
   zona davanti alla macchina (partenza robusta dalla mediana per riga);
2. ciò che sporge dal piano è **ostacolo** (rosso), il terreno libero è blu;
3. per 48 direzioni si misura fin dove il terreno resta libero e si sceglie la direzione con più spazio, penalizzando
   le sterzate inutili: **freccia + angolo + percentuale di spazio libero** (in metri se si inseriscono altezza e
   inclinazione della camera e campo visivo).

**Difetti trovati con foto reali e corretti** (gli errori più istruttivi del lavoro):
- Su una **facciata di palazzo** e su un **primo piano di pianta** l'app dichiarava «VIA LIBERA 100%»: il piano si
  adattava alla parete. Correzione: **controllo di plausibilità** sul gradiente di disparità (su un terreno cambia
  soprattutto in verticale; oltre 35° dalla verticale **non è terreno** → «TERRENO NON RICONOSCIUTO», nessuna freccia).
  Valori misurati: pavimento con libri 24°, pianta 60°, facciata 73°.
- Un **muro piatto e vicino** appiattisce la disparità e rompe il piano: ora dà «FERMARSI».
- Pannello «spazio libero» a 100% con terreno non riconosciuto: ora «--».
- Senza piano valido **non si dà mai una direzione** (una falsa sicurezza è peggio di nessuna risposta).

**Prove:** `test/generale.mjs` — 13 prove: scene sintetiche (libero, ostacolo a destra/sinistra/centro, muro, parete di
sbieco, telefono inclinato di 12°, camera calibrata) e **tre profondità vere** prodotte da MiDaS su foto reali.
Su un telefono reale è comparso (screenshot dell'utente) un caso di terreno non adatto: una stanza buia con un divano
disegnata come «corridoio» in modalità Vigna, a conferma che quel modello è specifico per l'erba tra i filari.

**Limiti dichiarati:** terreno piano e zona davanti alla macchina libera; distanze relative senza calibrazione, stima da
una sola immagine anche con calibrazione; **nessuna validazione sul terreno reale**; velocità non misurata
(550–800 ms/fotogramma in WebAssembly su una CPU di prova, sul telefono dipende dalla GPU).

---

## 6. Versione per Raspberry Pi (`raspberry/`)

Programma Python che gira **senza browser** (onnxruntime CPU + OpenCV).

- **`agrivision_rpi/nucleo.py` è generato dal notebook**, funzione per funzione, da `tools/genera_nucleo.py`
  (geometria, righello, cruscotto con marchio incorporato): il numero in cm è **quello del notebook**. Se esce una
  revisione nuova si rigenera; se una funzione è rinominata lo script si ferma e lo dice.
- **Sorgenti:** Raspberry Camera (CSI, `picamera2`), webcam USB (V4L2), file video, flussi `rtsp://`. Un thread legge e
  tiene **solo l'ultimo fotogramma**: il video resta in tempo reale anche con pochi fps.
- **Uscite:** pagina web MJPEG (`--stream 8080`, da telefono o PC sulla stessa rete, `/stato` in JSON), finestra
  (`--finestra`), video mp4 (`--registra`), registro json (formato del notebook).
- **Installazione:** `./installa.sh` (venv, `numpy<2` per compatibilità con `python3-opencv`); `./installa.sh servizio`
  avvia tutto all'accensione (systemd).
- **Opzioni:** `--sorgente`, `--corridoio-cm`, `--larghezza`, `--thread`, `--stile linee|area`, `--max-fotogrammi`, …
- **Prove:** `test_fumo.sh` (modello **finto** + video sintetico), `test_curve.py`.

**Requisiti:** Pi 4 (4 GB) o Pi 5, sistema **64 bit** (Bookworm o successivo). **Le prestazioni sulla scheda non sono
state misurate** (nessuna Raspberry disponibile): si misurano con `--max-fotogrammi 100` leggendo «rete … ms · … fps».
La Raspberry legge anche l'H.265 (ffmpeg), il browser desktop spesso no.

Uso tipico:
```bash
git clone https://github.com/NazBell01/Supporto_Guida_AgriVision.git
cd Supporto_Guida_AgriVision/raspberry && ./installa.sh
# copia drivable_migliore.onnx come guida.onnx in questa cartella
.venv/bin/python -m agrivision_rpi --modello guida.onnx --sorgente picamera --stream 8080
```

---

## 7. Stile «linee guida» (assistente di linea da retrocamera)

Richiesta: al posto del trapezio con griglia, due curve laterali e una **mezzeria di colore diverso** che curvano
secondo il sentiero. Implementato in JavaScript (`geometria.js`/`disegno.js`) **e** in Python (`raspberry/.../linee.py`).

**Aspetto:** due curve laterali azzurre che si assottigliano con la distanza, mezzeria **bianca tratteggiata**
(bianco perché il verde si perderebbe sull'erba), tacche trasversali come le barre di distanza, contorno scuro.
Righello e asse della macchina restano. Si sceglie nelle impostazioni (telefono) o con `--stile` (Raspberry);
`area` è il vecchio trapezio. Esempio: `docs/confronto_stili_guida.png` (scena **sintetica**, non una ripresa reale).

**Storia dei difetti e delle correzioni** (tutti emersi dalle prove e dai video reali dell'utente):

| Versione | Idea | Problema trovato | Correzione |
|---|---|---|---|
| 1 | Una parabola per ciascun bordo, ancorata alla retta del notebook | Il termine quadratico era quasi vuoto: la retta ai minimi quadrati assorbe già la curva (curvatura stimata 0,0001 contro 0,004 vera) | Adattamento quadratico libero ai bordi |
| 2 | Due parabole indipendenti | **Le curve si incrociavano** (35% delle maschere sporche, larghezza fino a −2460 px): un morso di fogliame su un lato piegava un bordo oltre l'altro | **Nastro:** una linea centrale che flette + semilarghezza in prospettiva; bordi = centro ∓ semilarghezza |
| 3 | Nastro con vincolo «tagliato» (clip) | **Spigolo in lontananza**: nel punto di saturazione del clip la linea «si impennava» di colpo (visto nel video di VignaStefano); spigoli >25° nel 54% delle maschere sporche | Il vincolo **scala** l'intera deviazione dalla retta invece di tagliarla; linee non disegnate oltre il punto di fuga; nastro scartato (→ dritto) se la larghezza stimata è in forte disaccordo con la retta |

**Modello finale:** `centro(y) = c0 + c1·u + c2·u²`, `semilarghezza(y) = h0 + h1·u` (u = y − yref), adattati ai punti
medi/semilarghezze per riga con scarto dei punti lontani, partendo dalle rette del notebook; vincolati (per scala) a
semilarghezza fra 0,6 e 1,4 volte quella della retta e centro entro mezza semilarghezza; **lisciati nel tempo** (media
esponenziale, peso 0,35) perché la traiettoria non salti da un fotogramma all'altro; ricade su un nastro dritto se la
maschera è povera.

**Risultati misurati (600 maschere sporche, JS e Python):** incroci **0%** (erano 35%), spigoli >25° **0%** (erano
54%), svolta massima media fra segmenti 35,6° → ~1,6°; sui sentieri puliti la curvatura vera è ritrovata (0,00398 su
0,00400, bordi entro 0,5 px). **Lo scarto in cm resta sempre quello del notebook:** le curve sono grafica, non una
misura nuova; dove il sentiero piega molto la retta con cui si misura è un'approssimazione.

---

## 8. Verifiche sui video reali dell'utente

### Registrazione dello zig zag (prima vigna, schermo dell'app con modello vero)
Lettura secondo per secondo (+ = corridoio a destra): −73, +23, +57, +39, −18, n/d, n/d, −31, ~10, +48, +74, +9, −22,
n/d, n/d, −59, ~6, +15, +28 cm. Il segno si inverte più volte (periodo ~7–8 s, ampiezza ±60–75 cm), 5 secondi su 19
(~26%) «misura inaffidabile» nei passaggi da un lato all'altro. **Interpretazione:** con il telefono in mano, ruotare la
camera sposta l'immagine come uno spostamento laterale: il numero in cm è lo **spostamento apparente** del corridoio
nell'immagine, non la distanza vera dal centro. Per sterzare una macchina con la camera fissa al telaio è coerente;
come misura in cm per la rotazione **non è validato**.
Scopo dichiarato dall'utente: lo zig zag serve a **capire i limiti della detection** ed evitare percorsi solo
rettilinei. (Nella conversazione non risultava traccia dell'accordo: lo si è assunto come obiettivo del test.)

### Seconda vigna (VignaStefano, ripresa verticale)
Percorso più stabile; difetto: in alcuni fotogrammi le linee «si impennavano» in alto → spigolo da clip (§7, versione 3).
**Una ripresa verticale dà al modello una striscia di soli ~108 px di larghezza** (il modello lavora su 320×192): per la
massima precisione si filma **in orizzontale**. Per VignaStefano la larghezza del corridoio **non è misurata**
(`CORRIDOIO_CM_PER_VIGNA` = None nel notebook): senza il metro il numero in cm non è fisico.

### Schermata nera con un video: HEVC
Il messaggio diagnostico dell'app indicava **0×0**: il browser non decodificava il video (tipicamente **H.265/HEVC**
di telefoni/action cam). Ora l'app lo verifica prima di partire e spiega come convertire (HandBrake, VLC, `ffmpeg -i
in -c:v libx264 -pix_fmt yuv420p -an out.mp4`).

---

## 9. Qualità, test e strumenti

| Prova | Cosa verifica |
|---|---|
| `test/confronta.mjs` | geometria JS = funzioni **originali Python** del notebook (9 maschere + righello) |
| `test/generale.mjs` | modalità generale: 13 prove (sintetiche + 3 profondità vere di MiDaS) |
| `test/curve.mjs` | nastro: fedeltà su sentieri puliti, **0 incroci** e **0 spigoli** su 600 maschere sporche |
| `raspberry/test_curve.py` | gli stessi controlli per la versione Python |
| `raspberry/test_fumo.sh` | Raspberry end-to-end con modello finto e video sintetico (89 fotogrammi, scarto −24,4…+24,9 cm) |
| Chromium + Playwright | app completa con videocamera simulata, video grandi (Full HD, 4K, verticali), offline con service worker |
| `.github/workflows/pages.yml` | esegue le prove e pubblica su Pages a ogni push su `main` |

Generatori di video sintetici in `test/`: `modello_finto.py`, `video_finto.py`, `video_curva.py`, `video_zigzag.py`.
Cross-check: stesso modello finto e stesso video danno gli **stessi scarti** in JavaScript e in Python.

**Cosa le prove NON dicono:** con un modello finto verificano pipeline, cruscotto e geometria, **non** la qualità del
modello vero né il comportamento delle linee sul fogliame reale.

---

## 10. Decisioni prese e perché

- **Repository separato e pubblico** per i test (la decisione sulla privacy dei pesi è rimandata). Con repo pubblico
  `guida.onnx` **non** si carica; `profondita.onnx` sì (MIT, ridistribuibile).
- **Nessuna scorciatoia sui numeri:** lo scarto in cm resta calcolato come nel notebook; il disegno non lo sostituisce.
- **Se non si sa, non si dice:** niente direzione senza terreno riconosciuto; niente cm senza scala; «misura
  inaffidabile» invece di un valore verosimile. Stessa filosofia del notebook (niente bussola/velocità/sensori finti).
- **Ausilio all'operatore, non un comando:** per muovere una macchina servono sensori ridondanti e un arresto
  indipendente dal telefono/Raspberry.
- Hugging Face non era raggiungibile dall'ambiente di lavoro: i modelli scaricati vengono da GitHub Releases
  (MiDaS: `isl-org/MiDaS`, v2_1).

## 11. Cose da sapere / limiti aperti

1. **Modello D-20:** gli errori 4,2/8,2 cm valgono per quella sessione; con un'altra run vanno aggiornati in
   `js/config.js` (e nel notebook).
2. **Camera:** i cm valgono con la camera montata **come nelle riprese di addestramento** (altezza, inclinazione,
   orizzontale). Un telefono in mano che oscilla cambia il significato del numero (vedi §8).
3. **Orientamento:** filmare in **orizzontale**.
4. **Scala per vigna:** 200 cm = Vigna94. Per le altre va misurata col metro.
5. **Prestazioni:** sul telefono sono state viste ~275 ms/fotogramma (modalità generale, WebAssembly, ~3 fps). Per la
   modalità Vigna e per la Raspberry **non ci sono misure reali**.
6. **Modalità generale** non è validata su terreni veri (erba, sentieri, cortili): è una prova.
7. **Privacy:** repository pubblico → chiunque può scaricare `profondita.onnx` (ok, MIT) e il codice; `guida.onnx` no.
8. **HEVC:** i video H.265 vanno convertiti per il browser; la Raspberry li legge direttamente.
9. **Pages:** richiede il repository con Source «GitHub Actions»; già attivo. Per repository privati serve un piano
   che lo consenta.
10. **Permessi:** l'app «Claude» di GitHub deve avere accesso al repository (già concesso a `Supporto_Guida_AgriVision`).

## 12. Prossimi passi consigliati

1. **Rifare la registrazione** con lo stesso filmato di VignaStefano dopo le ultime correzioni (§7, versione 3) e
   verificare incroci e spigoli sul fogliame vero; se restano difetti: accorciare ancora le linee o lisciarle di più.
2. Valutare di **ancorare le linee al fondo dell'immagine** (dove sta la macchina) e far fletterne solo la parte in
   lontananza, se è questo che si intende con «slittamento».
3. **Misurare la velocità** sul telefono e sulla Raspberry reali (riga diagnostica «provider · ms · fps»).
4. **Misurare la larghezza del corridoio** di VignaStefano (tre misure col metro) per avere cm fisici.
5. **Modalità generale:** aggiungere un **rilevatore di oggetti** (persone, pali) e valutare un modello di **spazio
   libero** addestrato su dati propri; **calibrare la camera** per distanze in metri; per sterzare davvero,
   odometria o SLAM (ARCore/ARKit solo su app nativa).
6. Decidere se tenere `guida.onnx` fuori dal repository pubblico (selettore a ogni apertura) o passare a un repository
   privato con Pages privato.
7. Collegare, in un passo a parte e con le dovute sicurezze, l'uscita del programma a uno sterzo.

---

## 13. Cronologia sintetica

| Fase | Che cosa è successo |
|---|---|
| 1 | Lettura del notebook REV_R; porting JS (geometria + cruscotto + onnxruntime-web); test contro il Python originale; PWA offline |
| 2 | Spostamento nel repository `Supporto_Guida_AgriVision`; permessi dell'app GitHub; attivazione di Pages; prima prova dell'utente da telefono: funziona |
| 3 | Modalità generale con MiDaS; correzione dei falsi «via libera» (facciata, pianta, muro) |
| 4 | Versione Raspberry: nucleo generato dal notebook, sorgenti, stream MJPEG, servizio systemd |
| 5 | Stile a linee guida; immagine di confronto; recupero del file `drivable_migliore.onnx` da Drive |
| 6 | Prove con video reali: schermata nera (HEVC, 0×0), messaggi diagnostici, WebGPU→WebAssembly |
| 7 | Zig zag: analisi della registrazione; incroci delle curve (35% → 0%) con il modello a **nastro** |
| 8 | VignaStefano: spigolo in lontananza (54% → 0%) con vincolo per **scala** e taglio al punto di fuga |

*Documento generato alla fine della sessione di lavoro del 14 ottobre 2026.*
