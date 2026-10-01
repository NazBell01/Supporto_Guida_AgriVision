# Modello di guida

L'app cerca qui il file **`guida.onnx`**: la rete di segmentazione del corridoio
(DeepLabV3 · MobileNetV3-Large, 2 classi, ingresso 320×192, normalizzazione ImageNet).
Il file **non è nel repository** perché i pesi stanno sul tuo Drive.

## Come ottenerlo

Il notebook REV_R lo produce già: la cella **1bis** esporta `drivable_migliore.pth` in ONNX,
verifica che dia le stesse maschere di PyTorch e lo salva accanto ai pesi:

```
MyDrive/07_AI_os/a3_Run/<run>/drivable_migliore.onnx
```

1. Scarica quel file da Drive.
2. Rinominalo **`guida.onnx`** e mettilo in questa cartella (`modelli/`).
   Pesa circa 40–45 MB: sotto il limite di 100 MB di GitHub, quindi basta un commit normale.
3. Il file `.gitignore` esclude `modelli/*.onnx` per evitare di pubblicare i pesi per sbaglio: se decidi di
   tenerlo nel repository, togli quella riga dal `.gitignore`.
4. Fai push su `main`: il workflow pubblica la pagina.

Se il file manca, l'app lo chiede all'avvio con un selettore di file: va bene per provare dal
telefono senza pubblicare nulla (resta in memoria, va riscelto a ogni apertura).

## Attenzione

- **Pubblico = pubblico.** Con un repository pubblico, chiunque può scaricare `guida.onnx`.
  Se i pesi non devono circolare, usa un repository privato con un piano che include Pages privato,
  oppure tieni il file fuori dal repo e usa il selettore.
- **Sessione.** `js/config.js` riporta gli errori misurati sulla **D-20** (4,2 cm / 8,2 cm).
  Se esporti una run diversa, aggiorna `sessione` e le due cifre dalla `scheda_sessione.json`:
  il cruscotto usa 8,2 cm per la banda di incertezza sul righello.
- **Ingresso fisso.** Il modello è esportato con ingresso 1×3×192×320; l'app lo verifica e lo dice
  se non coincide.

## profondita.onnx (modalità generale)

Incluso nel repository: è MiDaS small v2.1, licenza MIT (vedi `MiDaS-LICENSE.txt`), 67 MB, ingresso 1×3×256×256.
Non va esportato da nessuna parte. Un'alternativa più precisa ma troppo pesante per un telefono in tempo reale
è Depth Anything V2 (518×518, 99 MB).
