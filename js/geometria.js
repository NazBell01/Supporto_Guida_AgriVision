// Porting JavaScript delle funzioni di geometria del notebook REV_R.
// Devono restare IDENTICHE a quelle del notebook di addestramento:
// scarto_laterale, geometria_corridoio (dalla D-16), _retta_robusta, Righello.
// Nessuna dipendenza dal DOM: si può provare con node (vedi test/).

export const GUIDA_DIM = [320, 192];          // [L, A] deve combaciare con IMAGE_SIZE dell'addestramento
export const MEDIA_IMGNET = [0.485, 0.456, 0.406];
export const DEV_IMGNET   = [0.229, 0.224, 0.225];

// ---------------------------------------------------------------- utilità ---
function polyfit1(y, x) {                      // minimi quadrati, retta x = a*y + b
  const n = y.length;
  let sy = 0, sx = 0, syy = 0, syx = 0;
  for (let i = 0; i < n; i++) { sy += y[i]; sx += x[i]; syy += y[i]*y[i]; syx += y[i]*x[i]; }
  const den = n*syy - sy*sy;
  const a = den === 0 ? 0 : (n*syx - sy*sx) / den;
  return [a, (sx - a*sy) / n];
}
export const polyval = (p, y) => p[0]*y + p[1];

export function mediana(v) {
  if (!v.length) return NaN;
  const s = Float64Array.from(v).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
}

// ------------------------------------------------------ preelaborazione ----
// Lettera di casella: conserva le proporzioni, poi normalizzazione ImageNet.
// Restituisce il riquadro utile (ox, oy, nw, nh) sulla tela L×A.
export function rettangoloUtile(w, h, dim = GUIDA_DIM) {
  const [L, A] = dim;
  const s = Math.min(L / w, A / h);
  const nw = Math.max(1, Math.round(w * s));
  const nh = Math.max(1, Math.round(h * s));
  return { ox: Math.floor((L - nw) / 2), oy: Math.floor((A - nh) / 2), nw, nh };
}

// rgba: Uint8ClampedArray L*A*4 (tela già con lettera di casella) -> Float32 NCHW 1×3×A×L
export function normalizza(rgba, dim = GUIDA_DIM, out = null) {
  const [L, A] = dim, n = L * A;
  out = out || new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const j = i * 4;
    out[i]         = (rgba[j]     / 255 - MEDIA_IMGNET[0]) / DEV_IMGNET[0];
    out[n + i]     = (rgba[j + 1] / 255 - MEDIA_IMGNET[1]) / DEV_IMGNET[1];
    out[2 * n + i] = (rgba[j + 2] / 255 - MEDIA_IMGNET[2]) / DEV_IMGNET[2];
  }
  return out;
}

// logits: Float32 [2, A, L] -> maschera Uint8 A*L (1 = percorribile, classe 1)
export function maschera(logits, dim = GUIDA_DIM) {
  const [L, A] = dim, n = L * A;
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = logits[n + i] > logits[i] ? 1 : 0;
  return m;
}

// ------------------------------------------------------- scarto laterale ---
// Centro orizzontale della regione percorribile nella fascia bassa, in pixel
// rispetto al centro. Positivo = corridoio a destra.
export function scartoLaterale(m, L, A, frazioneBassa = 0.35) {
  const y0 = Math.trunc(A * (1 - frazioneBassa));
  let tot = 0, somma = 0;
  for (let x = 0; x < L; x++) {
    let p = 0;
    for (let y = y0; y < A; y++) p += m[y * L + x];
    tot += p; somma += p * x;
  }
  if (tot < 1) return null;
  return somma / tot - (L - 1) / 2;
}

// ---------------------------------------------------- geometria corridoio --
function rettaRobusta(y, x, giri = 3) {
  let p = polyfit1(y, x);
  for (let g = 0; g < giri; g++) {
    const r = x.map((xi, i) => Math.abs(xi - polyval(p, y[i])));
    const s = Math.max(mediana(r), 0.5);
    const yk = [], xk = [];
    for (let i = 0; i < r.length; i++) if (r[i] < 4 * s) { yk.push(y[i]); xk.push(x[i]); }
    if (yk.length < 8) break;
    p = polyfit1(yk, xk);
  }
  return p;
}

// Ritorna {centro, larghezza, rette:[pl,pr], yAlto, yBasso} oppure null.
export function geometriaCorridoio(m, L, A, margineAlto = 0.05, margineBordo = 0.02) {
  const sx = new Int32Array(A).fill(-1), dx = new Int32Array(A).fill(-1);
  const presenti = [];
  for (let y = 0; y < A; y++) {
    let a = -1, b = -1;
    const r = y * L;
    for (let x = 0; x < L; x++) if (m[r + x]) { a = x; break; }
    if (a < 0) continue;
    for (let x = L - 1; x >= 0; x--) if (m[r + x]) { b = x; break; }
    sx[y] = a; dx[y] = b; presenti.push(y);
  }
  if (presenti.length < 20) return null;
  const ap = presenti[0];
  const lim = Math.max(1.0, margineBordo * L);
  const uso = presenti.filter(y => ap + margineAlto * A <= y && y <= A - 0.03 * A);
  const ls = uso.filter(y => sx[y] > lim);
  const ld = uso.filter(y => dx[y] < L - lim);
  if (ls.length < 10 || ld.length < 10) return null;
  const pl = rettaRobusta(ls, ls.map(y => sx[y]));
  const pr = rettaRobusta(ld, ld.map(y => dx[y]));
  const yref = 0.95 * A;
  const bs = polyval(pl, yref), bd = polyval(pr, yref);
  return {
    centro: (bs + bd) / 2 - (L - 1) / 2,
    larghezza: bd - bs,
    rette: [pl, pr],
    yAlto: presenti[0], yBasso: presenti[presenti.length - 1],
    bordi: { ls, ld, sx, dx },                    // righe usate e bordi per riga: servono alla curvatura
  };
}

// Mediana mobile della larghezza: riconosce una stima non credibile.
export class Righello {
  constructor(memoria = 45) { this.memoria = memoria; this.storia = []; }
  aggiorna(l) {
    if (l && l > 5) { this.storia.push(l); if (this.storia.length > this.memoria) this.storia.shift(); }
  }
  credibile(l) {
    if (!l || l <= 5) return false;
    if (this.storia.length < 8) return true;
    const med = mediana(this.storia);
    return 0.5 * med <= l && l <= 2.0 * med;
  }
}

// Dalle coordinate della tela a quelle del fotogramma (stesse formule del notebook).
export function rettePerFotogramma(rette, util, Wf, Hf) {
  const { ox, oy, nw, nh } = util;
  const sxf = Wf / nw, syf = Hf / nh;
  const porta = p => [p[0] / syf * sxf, (p[1] + p[0] * oy - ox) * sxf];
  return [porta(rette[0]), porta(rette[1])];
}

// Stato di una misura, come nel ciclo della cella 3.
export function statoMisura(misura, mascheraPresente, affidabile) {
  if (misura !== null && misura !== undefined) return 'ok';
  return (mascheraPresente && !affidabile) ? 'incerto' : 'assente';
}

// Un passo completo della guida su una maschera già calcolata.
// Ritorna {sc, geo, affidabile, cm} come guida() del notebook.
export function analizzaMaschera(m, righello, corridoioCm, dim = GUIDA_DIM) {
  const [L, A] = dim;
  const sc = scartoLaterale(m, L, A);
  const geo = geometriaCorridoio(m, L, A);
  const larghezza = geo ? geo.larghezza : null;
  const affidabile = righello.credibile(larghezza);
  righello.aggiorna(larghezza);
  let cm = null;
  if (sc !== null && affidabile && corridoioCm) cm = sc / larghezza * corridoioCm;
  return { sc, geo, affidabile, cm };
}

// Per la modalità generale (js/generale.js): stessa regressione robusta della guida a corridoio.
export { rettaRobusta };

// ----------------------------------------------------------- linee guida ----
// Per l'«assistente di linea» (stile retrocamera) i due bordi del corridoio diventano curve:
//     x(y) = c0 + c1·(y − yref) + c2·(y − yref)²
// adattate ai bordi per riga della maschera (minimi quadrati con scarto dei punti lontani, partendo
// dalla retta del notebook). Dove il corridoio è dritto c2 ≈ 0 e la curva coincide con la retta.
// Dove piega, la retta dello scarto in cm (identica al notebook) è un'approssimazione e la curva
// mostra la forma che il modello vede. Lo scarto in cm resta quello del notebook: la curva è
// grafica, non una misura nuova.
function risolvi3(M) {                       // sistema 3×3 [A|b] con eliminazione di Gauss; null se singolare
  for (let i = 0; i < 3; i++) {
    let m = i; for (let j = i + 1; j < 3; j++) if (Math.abs(M[j][i]) > Math.abs(M[m][i])) m = j;
    [M[i], M[m]] = [M[m], M[i]];
    if (Math.abs(M[i][i]) < 1e-9) return null;
    for (let j = i + 1; j < 3; j++) { const f = M[j][i] / M[i][i]; for (let k = i; k < 4; k++) M[j][k] -= f * M[i][k]; }
  }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) { let t = M[i][3]; for (let k = i + 1; k < 3; k++) t -= M[i][k] * x[k]; x[i] = t / M[i][i]; }
  return x;
}

export function curveBordi(bordi, rette, yrefT) {
  return [[bordi.ls, bordi.sx, rette[0]], [bordi.ld, bordi.dx, rette[1]]].map(([ys, bordo, p]) => {
    const u = ys.map(y => y - yrefT), e = ys.map(y => bordo[y]);
    let c = [polyval(p, yrefT), p[0], 0];
    for (let giro = 0; giro < 4; giro++) {
      const res = e.map((ei, i) => ei - (c[0] + c[1] * u[i] + c[2] * u[i] * u[i]));
      const sg = Math.max(1.4826 * mediana(res.map(Math.abs)), 0.5);
      const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
      for (let i = 0; i < u.length; i++) if (Math.abs(res[i]) < 3 * sg) {
        const r = [1, u[i], u[i] * u[i]];
        for (let j = 0; j < 3; j++) { for (let k = 0; k < 3; k++) M[j][k] += r[j] * r[k]; M[j][3] += r[j] * e[i]; }
      }
      const n = risolvi3(M); if (!n) break; c = n;
    }
    return { c0: c[0], c1: c[1], c2: c[2] };
  });
}

// Limita la curvatura: nel tratto disegnato lo scostamento quadratico non supera `quota` volte la
// larghezza del corridoio alla riga di riferimento (oltre, è rumore della maschera).
export function limitaCurvatura(k, spanT, larghT, quota = 0.35) {
  const max = quota * larghT / Math.max(spanT * spanT, 1);
  return Math.max(-max, Math.min(max, k));
}

// Punti delle tre curve (sinistra, destra, mezzeria) in coordinate del fotogramma, dal basso verso l'alto.
// cv = [{c0, c1, c2}, {c0, c1, c2}] nelle coordinate della tela del modello; yrefT è la riga di riferimento
// usata per ricavarle (lo stesso valore passato a curveBordi).
export function curveFotogramma(cv, util, Wf, Hf, yrefT, yBassoF, yAltoF, n = 30) {
  const { ox, oy, nw, nh } = util, sxf = Wf / nw, syf = Hf / nh;    // yrefT: riga di riferimento della tela, quella di curveBordi
  const out = { sx: [], dx: [], mid: [] };
  const val = (c, u) => (c.c0 + c.c1 * u + c.c2 * u * u - ox) * sxf;
  for (let i = 0; i < n; i++) {
    const yf = yBassoF + (yAltoF - yBassoF) * (n > 1 ? i / (n - 1) : 0), u = yf / syf + oy - yrefT;
    const xl = val(cv[0], u), xr = val(cv[1], u);
    out.sx.push([xl, yf]); out.dx.push([xr, yf]); out.mid.push([(xl + xr) / 2, yf]);
  }
  return out;
}
