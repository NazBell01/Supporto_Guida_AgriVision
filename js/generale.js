// MODALITÀ GENERALE: ostacoli e direzione consigliata, senza sapere che si è in vigna.
//
// Principio (v-disparità): su un terreno piano la disparità d = 1/Z cresce LINEARMENTE con
// la riga dell'immagine v, d = a·v + b, perché Z = f·h/(v − v0). MiDaS dà una disparità
// relativa (scala e traslazione ignote), ma resta lineare in v. Quindi:
//   1. si adatta una retta d(v) alla zona davanti alla macchina (si assume terreno libero);
//   2. un pixel con disparità MAGGIORE di quella del terreno a quella riga sporge dal piano:
//      è più vicino di quanto sarebbe il suolo, cioè un ostacolo;
//   3. per ogni direzione si guarda fin dove il terreno resta libero, e si sceglie la direzione
//      con più spazio, penalizzando le sterzate inutili.
// Nessun addestramento. Limiti: terreno ~piano, zona davanti alla macchina libera all'inizio,
// distanze RELATIVE se la camera non è calibrata (altezza, inclinazione, campo visivo).

import { rettaRobusta, mediana } from './geometria.js';

export const PARAM = {
  nBin: 48,               // direzioni in cui si divide il campo visivo
  zonaTerreno: { y0: 0.62, y1: 0.97, x0: 0.25, x1: 0.75 },   // dove si adatta il piano (frazioni)
  ignoraBasso: 0.03,      // riga più bassa: il cofano della macchina
  kSoglia: 4.0,           // scarti del terreno oltre i quali un pixel è ostacolo (× MAD)
  sogliaMin: 0.07,        // …e comunque almeno questa differenza (disparità normalizzata 0–1)
  frazioneBin: 0.40,      // quota di colonne del bin con ostacolo perché il bin sia bloccato
  righeConsecutive: 2,
  hfovGradi: 65,          // campo visivo orizzontale: tipico di un telefono; si può calibrare
  larghMacchinaGradi: 24, // ingombro angolare della macchina davanti a sé
  sogliaFermoRel: 0.12,   // spazio libero relativo sotto cui si consiglia di fermarsi
  sogliaFermoM: 1.5,      // idem in metri, se la camera è calibrata
  sogliaDrittoGradi: 6,
  angoloMaxGradi: 35,     // inclinazione massima del gradiente di disparità rispetto alla verticale
  penaleSterzo: 0.20, penalePrecedente: 0.25,
};

// Disparità grezza -> 0..1 con percentili 2-98 (scala e traslazione di MiDaS cambiano a ogni fotogramma).
export function normalizza(d) {
  const n = d.length, s = new Float32Array(n);
  const campione = []; for (let i = 0; i < n; i += 7) campione.push(d[i]);
  campione.sort((p, q) => p - q);
  const lo = campione[Math.floor(campione.length * 0.02)], hi = campione[Math.floor(campione.length * 0.98)];
  const den = hi - lo;
  if (!(den > 1e-9)) return { d: s, valida: false };
  for (let i = 0; i < n; i++) s[i] = Math.min(1, Math.max(0, (d[i] - lo) / den));
  return { d: s, valida: true };
}

// Conversione riga -> distanza in metri, con camera calibrata. null se non calibrata.
export function distanzaDaRiga(v, H, cal) {
  if (!cal || !(cal.altezzaM > 0)) return null;
  const f = (H / 2) / Math.tan((cal.vfovGradi * Math.PI / 180) / 2);
  const ang = (cal.inclinazioneGradi * Math.PI / 180) + Math.atan((v - H / 2) / f);   // sotto l'orizzonte
  if (ang <= 0.01) return Infinity;
  return Math.min(60, cal.altezzaM / Math.tan(ang));
}

// d: Float32 normalizzato W×H. Ritorna il piano del terreno, la maschera degli ostacoli e lo spazio per direzione.
export function analizza(d, W, H, prec = null, cal = null, p = PARAM) {
  // 1. piano del terreno d = a·v + c·x + b nella zona davanti alla macchina (minimi quadrati
  //    robusti). Il termine c·x tiene conto di un telefono inclinato di lato.
  const z = p.zonaTerreno, y0 = Math.floor(z.y0 * H), y1 = Math.floor(z.y1 * H);
  const x0 = Math.floor(z.x0 * W), x1 = Math.floor(z.x1 * W);
  const PY = [], PX = [], PD = [];
  for (let y = y0; y < y1; y += 2) for (let x = x0; x < x1; x += 2) { PY.push(y); PX.push(x); PD.push(d[y * W + x]); }
  // partenza robusta: mediana per riga (un ostacolo che occupa meno della metà delle colonne non la sposta)
  const vv = [], dm = [];
  for (let y = y0; y < y1; y += 2) {
    const r = []; for (let x = x0; x < x1; x += 2) r.push(d[y * W + x]);
    vv.push(y); dm.push(mediana(r));
  }
  const p0 = rettaRobusta(vv, dm, 4);
  let coef = [p0[0], 0, p0[1]];
  let uso = null, mad = 0;
  for (let giro = 0; giro < 4; giro++) {
    const res = PD.map((di, i) => di - (coef[0] * PY[i] + coef[1] * PX[i] + coef[2]));
    mad = 1.4826 * mediana(res.map(Math.abs));
    uso = res.map(r => Math.abs(r) < 4 * Math.max(mad, 0.01));
    coef = adattaPiano(PY, PX, PD, uso);
  }
  const [a, c, b] = coef;
  const dentro = uso.filter(Boolean).length / uso.length;
  // Controllo di plausibilità: su un terreno la disparità cambia soprattutto verso l'alto/basso
  // dell'immagine; su una parete vista di sbieco o su un primo piano cambia soprattutto di lato.
  // Se la direzione del gradiente si allontana troppo dalla verticale, quello NON è il terreno.
  const angolo = Math.atan2(Math.abs(c), a) * 180 / Math.PI;
  const pianoOk = a > 0.0005 && dentro > 0.45 && angolo < p.angoloMaxGradi;
  const vOriz = pianoOk ? -(b + c * W / 2) / a : 0;          // riga (al centro) dove il terreno prevede disparità 0
  const soglia = Math.max(p.sogliaMin, p.kSoglia * mad);
  const dd = PD;
  const pr = [a, c, b];
  const prevedi = (x, y) => a * y + c * x + b;

  // 2. ostacoli: disparità sopra il piano
  const ost = new Uint8Array(W * H);
  if (pianoOk) {
    const yMax = Math.floor((1 - p.ignoraBasso) * H);
    for (let y = 0; y < yMax; y++) {
      for (let x = 0; x < W; x++) if (d[y * W + x] - prevedi(x, y) > soglia) ost[y * W + x] = 1;
    }
  }

  // 3. spazio libero per direzione: da quale riga, salendo dal basso, il bin è bloccato
  const nB = p.nBin, yBasso = Math.floor((1 - p.ignoraBasso) * H) - 1;
  const yTop = Math.max(0, Math.ceil(pianoOk ? Math.min(vOriz, H * 0.45) : H * 0.2));   // oltre l'orizzonte non c'è terreno
  const vLibero = new Float32Array(nB).fill(yTop);
  const bloccato = new Uint8Array(nB);
  for (let i = 0; i < nB; i++) {
    const xa = Math.floor(i * W / nB), xb = Math.floor((i + 1) * W / nB);
    let run = 0;
    for (let y = yBasso; y >= yTop; y--) {
      let c = 0; for (let x = xa; x < xb; x++) c += ost[y * W + x];
      if (c / (xb - xa) >= p.frazioneBin) { run++; if (run >= p.righeConsecutive) { vLibero[i] = y + run; bloccato[i] = 1; break; } }
      else run = 0;
    }
  }
  // spazio relativo 0..1 (quanto del campo di terreno visibile resta libero) e in metri se calibrato
  const span = Math.max(1, yBasso - yTop);
  const libero = new Float32Array(nB), metri = cal ? new Float32Array(nB) : null;
  for (let i = 0; i < nB; i++) {
    libero[i] = Math.max(0, Math.min(1, (yBasso - vLibero[i]) / span));
    if (metri) metri[i] = distanzaDaRiga(vLibero[i], H, cal);
  }

  // 4. direzione consigliata
  const f = (W / 2) / Math.tan((p.hfovGradi * Math.PI / 180) / 2);
  const angBin = i => Math.atan(((i + 0.5) * W / nB - W / 2) / f) * 180 / Math.PI;
  const mezza = Math.max(1, Math.round((p.larghMacchinaGradi / p.hfovGradi * nB) / 2));
  let migliore = -1, punteggio = -Infinity, chiaroCentro = 1;
  const misura = i => (metri ? Math.min(...window(metri, i, mezza)) : Math.min(...window(libero, i, mezza)));
  for (let i = 0; i < nB; i++) {
    const m = misura(i);
    const norm = metri ? Math.min(1, m / 10) : m;         // 10 m = «tutto libero» per il punteggio
    const sc = norm - p.penaleSterzo * Math.abs(angBin(i)) / 90
                    - (prec !== null ? p.penalePrecedente * Math.abs(angBin(i) - prec) / 90 : 0);
    if (sc > punteggio) { punteggio = sc; migliore = i; }
  }
  const centro = Math.round(nB / 2 - 0.5);
  chiaroCentro = misura(centro);
  const chiaroMigliore = misura(migliore);
  const rotta = angBin(migliore);
  const fermoSoglia = metri ? p.sogliaFermoM : p.sogliaFermoRel;
  const fermo = pianoOk && chiaroMigliore < fermoSoglia;

  // Senza un piano credibile NON si dà una direzione: «libero» sarebbe una falsa sicurezza.
  // Caso particolare: una superficie vicina che riempie la zona davanti (un muro) appiattisce la
  // disparità e rompe il piano; è proprio la situazione in cui bisogna fermarsi.
  const medZona = mediana(dd);
  let azione = 'DRITTO';
  if (!pianoOk) azione = (!(a > 0.0005) && Math.abs(c) < 0.0005 && medZona > 0.6 && dentro > 0.45) ? 'FERMO' : null;   // solo superficie piatta e vicina
  else if (fermo) azione = 'FERMO';
  else if (rotta > p.sogliaDrittoGradi) azione = 'DESTRA';
  else if (rotta < -p.sogliaDrittoGradi) azione = 'SINISTRA';

  return { pianoOk, angolo, medZona, piano: pr, vOriz, mad, soglia, dentro, ostacoli: ost, vLibero, libero, metri, bloccato,
           rotta, azione, chiaroCentro, chiaroMigliore, migliorBin: migliore, angBin, nB, yBasso, yTop };
}

function window(a, i, mezza) {
  const out = [];
  for (let j = Math.max(0, i - mezza); j <= Math.min(a.length - 1, i + mezza); j++) out.push(a[j]);
  return out;
}

// Minimi quadrati per d = a·y + c·x + b (equazioni normali 3×3, eliminazione di Gauss).
function adattaPiano(Y, X, D, uso) {
  const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < Y.length; i++) {
    if (uso && !uso[i]) continue;
    const r = [Y[i], X[i], 1];
    for (let j = 0; j < 3; j++) { for (let k = 0; k < 3; k++) M[j][k] += r[j] * r[k]; M[j][3] += r[j] * D[i]; }
  }
  for (let i = 0; i < 3; i++) {
    let m = i; for (let j = i + 1; j < 3; j++) if (Math.abs(M[j][i]) > Math.abs(M[m][i])) m = j;
    [M[i], M[m]] = [M[m], M[i]];
    if (Math.abs(M[i][i]) < 1e-12) return [0, 0, 0];
    for (let j = i + 1; j < 3; j++) { const f = M[j][i] / M[i][i]; for (let k = i; k < 4; k++) M[j][k] -= f * M[i][k]; }
  }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) { let t = M[i][3]; for (let k = i + 1; k < 3; k++) t -= M[i][k] * x[k]; x[i] = t / M[i][i]; }
  return x;
}
