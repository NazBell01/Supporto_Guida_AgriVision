// Parametri del simulatore, dalla cella 1 del notebook REV_R.
export const CONFIG = {
  modello: 'modelli/guida.onnx',   // esportato dalla cella 1bis del notebook (vedi modelli/LEGGIMI.md)
  logo: 'logo.png',                // facoltativo: se manca, il cruscotto scrive il nome

  // Larghezza reale del corridoio erboso, misurata col metro. null = «scala non
  // misurata»: lo scarto si mostra in pixel e il cruscotto lo dichiara. 200 cm è il
  // metro di Vigna94; per un'altra vigna va cambiato nelle impostazioni dell'app.
  corridoioCm: 200,
  soglia: 15,                      // sotto questo scarto (cm) l'azione è DRITTO

  // Errori del modello, MISURATI sulla sessione D-20 (06/09/2026). Il secondo, su una
  // giornata diversa, disegna la banda di incertezza sul righello. Se i pesi esportati
  // vengono da un'altra run, questi numeri valgono come riferimento, non come misura.
  sessione: 'D-20',
  erroreDentroCm: 4.2,
  erroreFuoriCm: 8.2,

  secondiTraccia: 20,
};
