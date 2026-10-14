# AgriVision su Raspberry Pi

Il simulatore di guida (notebook **REV_R**) come programma Python che gira sulla Raspberry, senza
browser: legge la videocamera, calcola scarto laterale e larghezza del corridoio con il modello di
guida e disegna il **cruscotto Kydo** sul video. Lo vedi in una pagina web (da telefono o PC sulla
stessa rete), su uno schermo collegato, o lo registri in un file.

Lo stile delle linee guida (`agrivision_rpi/linee.py`) è l'unica parte aggiunta al disegno: curve per i due bordi e una mezzeria, ancorate alla stessa retta del notebook. Lo scarto in cm resta quello del notebook.

La geometria, il righello e il cruscotto **non sono riscritti**: `agrivision_rpi/nucleo.py` è generato
dal notebook, funzione per funzione (`tools/genera_nucleo.py`). Il numero in centimetri è quello del
notebook.

## Cosa serve

- **Raspberry Pi 4 (4 GB) o Pi 5**, sistema **64 bit** (Raspberry Pi OS Bookworm o successivo).
  Un Pi 3 o un sistema a 32 bit non vanno bene per onnxruntime.
- Una videocamera: **Raspberry Camera** (CSI) o una **webcam USB**.
- Il file **`guida.onnx`**: è `drivable_migliore.onnx` che la cella 1bis del notebook ha già scritto accanto ai pesi su
  Drive (`07_AI_os/a3_Run/<run>/`). Copialo sulla Raspberry con una chiavetta USB o con `scp`.
  Non sta nel repository.

## Installazione

Sulla Raspberry, con la rete collegata:

```bash
git clone https://github.com/NazBell01/Supporto_Guida_AgriVision.git
cd Supporto_Guida_AgriVision/raspberry
./installa.sh
```

(Senza `git`: dalla pagina del repository **Code → Download ZIP**, copia e decomprimi.)
Copia il modello in questa cartella: `cp /percorso/drivable_migliore.onnx guida.onnx`.

## Avvio

```bash
# videocamera Raspberry (CSI), video in rete sulla porta 8080
.venv/bin/python -m agrivision_rpi --modello guida.onnx --sorgente picamera --stream 8080

# webcam USB
.venv/bin/python -m agrivision_rpi --modello guida.onnx --sorgente usb:0 --stream 8080

# prova con un filmato già girato (stessi numeri del notebook)
.venv/bin/python -m agrivision_rpi --modello guida.onnx --sorgente Test_Vigna94.mp4 --registra uscita.mp4

# con uno schermo collegato alla Raspberry: finestra, tasto q per uscire
.venv/bin/python -m agrivision_rpi --modello guida.onnx --sorgente picamera --finestra
```

Per vedere il video in rete: trova l'indirizzo con `hostname -I` e apri `http://<indirizzo>:8080/`
dal telefono o dal PC. `http://<indirizzo>:8080/stato` dà l'ultima misura in JSON.

Per fermare: `Ctrl+C`. Il registro degli scarti (stesso formato del notebook: chiave `scarti` e un
record per fotogramma) viene scritto accanto al video, o dove indichi con `--registro BASE`.

### Opzioni utili

| opzione | a cosa serve |
|---|---|
| `--corridoio-cm 200` | larghezza reale del corridoio, in cm. **200 vale per Vigna94**; per un'altra vigna va misurata. `0` = scala non misurata: lo scarto è in pixel e il cruscotto lo dichiara |
| `--stile linee` | `linee` (default): **assistente di linea**, con due curve laterali e una mezzeria tratteggiata che seguono il sentiero come le linee guida di una retrocamera, sotto forma di nastro che flette senza mai incrociarsi. `area`: trapezio e griglia del notebook |
| `--larghezza 960` | larghezza di lavoro. Più piccola = più veloce (la rete lavora comunque a 320×192; il costo è il disegno e la codifica) |
| `--thread N` | thread di onnxruntime (default: automatico) |
| `--registra file.mp4` | salva il video col cruscotto e il registro |
| `--max-fotogrammi 50` | si ferma dopo N fotogrammi: serve per misurare la velocità |

### Velocità: misurala sulla tua scheda

Non ho una Raspberry per misurarla: i tempi sulla scheda sono da verificare. Lancia
`--max-fotogrammi 100` e leggi le righe «rete … ms · … fps». La rete (DeepLabV3 MobileNetV3, 320×192) gira
solo su CPU. Se i fps sono pochi: `--larghezza 640`, un Pi 5, o più `--thread`. Il programma tiene **sempre l'ultimo
fotogramma** e scarta quelli arretrati: il video resta in tempo reale, ma con pochi fps la traccia dello scarto è
meno fitta.

## Avvio automatico all'accensione

```bash
MODELLO=$PWD/guida.onnx SORGENTE=picamera ./installa.sh servizio
systemctl status agrivision        # stato
journalctl -u agrivision -f        # log
sudo systemctl disable --now agrivision   # per toglierlo
```

Il servizio pubblica il video sulla porta 8080 e scrive i registri in `registri/`.

## Prove

`PYTHONPATH=. python3 test_curve.py` verifica le linee guida: fedeltà su sentieri puliti e, su 600 maschere sporche, che le due curve non si incrocino mai.

`./test_fumo.sh` (servono `onnx`, `onnxruntime`, `opencv`, `ffmpeg`) esegue tutto con un modello **finto** e un video
sintetico: verifica sorgente, cruscotto, video in rete e registro. **Non dice niente sulla qualità del modello.**

## Nuova revisione del notebook

```bash
python3 tools/genera_nucleo.py Simulatore_AgriVision_REV_X.ipynb agrivision_rpi/nucleo.py
```

Se una funzione è stata rinominata, lo script si ferma e dice quale.

## Limiti, come nel notebook

- **Ausilio all'operatore, non un comando alla macchina.** Il modello sbaglia 4,2 cm su corridoi mai visti della stessa
  giornata di addestramento e 8,2 cm su una giornata diversa (sessione D-20). Per muovere una macchina servono sensori e un
  arresto indipendenti.
- I centimetri valgono con **la camera montata come nelle riprese di addestramento** (altezza, inclinazione, orizzontale)
  e con la larghezza del corridoio di quella vigna.
- Il programma non comanda niente: legge il video e disegna. Collegarlo a uno sterzo è un passo a parte.
