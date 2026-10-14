// Prove dello stile «linee guida» (nastro: linea centrale che flette + semilarghezza in prospettiva):
//  1. fedeltà su sentieri puliti: ritrova la curvatura vera e i bordi cadono sulla maschera;
//  2. robustezza su maschere SPORCHE (morsi di fogliame, buchi, rumore): le due curve non si incrociano mai.
import { geometriaCorridoio, nastroBordi, puntiNastro } from '../js/geometria.js';
const L = 320, A = 192, yref = 0.95 * A;
let falliti = 0;
const prova = (n, ok, i = '') => { console.log(`${ok ? '✓' : '✗'} ${n} ${i}`); if (!ok) falliti++; };
let seed = 5; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;   // [-0.5, 0.5)
const norm = () => (rnd() + rnd() + rnd() + rnd()) * 1.7;                                  // ~gaussiana, sigma ≈ 1

// ---- 1. sentieri puliti ---------------------------------------------------------------------
function pulita(k0, rumore = 0) {
  const m = new Uint8Array(L * A);
  for (let y = 40; y < A; y++) {
    const t = (y - 40) / (A - 41), c = 160 + k0 * (y - yref) ** 2, h = 30 + 70 * t;
    const a = Math.max(0, Math.round(c - h + rumore * norm())), b = Math.min(L - 1, Math.round(c + h + rumore * norm()));
    for (let x = a; x <= b; x++) m[y * L + x] = 1;
  }
  return m;
}
const stima = m => { const g = geometriaCorridoio(m, L, A); return { g, par: nastroBordi(g.bordi, g.rette, yref) }; };
const v = 0.0040;
for (const [k0, nome] of [[0, 'dritto'], [v, 'da un lato'], [-v, "dall'altro lato"]]) {
  const m = pulita(k0), { g, par } = stima(m), c2 = par.c[2];
  const righe = []; for (let y = g.yAlto + 20; y < g.yBasso - 3; y += 5) righe.push(y);
  const q = puntiNastro(par, g.rette, righe, yref);
  let err = 0;
  righe.forEach((y, i) => {
    let a = 0; while (!m[y * L + a]) a++; let b = L - 1; while (!m[y * L + b]) b--;
    err = Math.max(err, Math.abs(q.xl[i] - a), Math.abs(q.xr[i] - b));
  });
  const ok = k0 === 0 ? Math.abs(c2) < 1e-4 : Math.abs(c2 - k0) < 0.3 * v;
  prova(`sentiero ${nome}: curvatura ${k0 === 0 ? '~0' : 'vicina alla vera'} e bordi sulla maschera`, ok && err < 5, `(c2 ${c2.toFixed(5)}, vera ${k0.toFixed(5)}; errore massimo ${err.toFixed(1)} px su 320)`);
}
{ const { par } = stima(pulita(v, 1.4)); prova('con rumore la curvatura resta riconoscibile', Math.abs(par.c[2] - v) < 0.45 * v, `(${par.c[2].toFixed(5)})`); }

// ---- 2. maschere sporche --------------------------------------------------------------------
function sporca() {
  const m = new Uint8Array(L * A), yaw = (rnd() * 2) * 40, top = Math.round(43 + 22 * (rnd() + 0.5));
  for (let y = top; y < A; y++) {
    const t = (y - top) / (A - top), c = 160 + yaw * (1 - t) * 0.6, h = 20 + 100 * t;
    const a = Math.max(0, Math.round(c - h + norm())), b = Math.min(L - 1, Math.round(c + h + norm()));
    for (let x = a; x <= b; x++) m[y * L + x] = 1;
  }
  const morsi = 1 + Math.floor((rnd() + 0.5) * 4);
  for (let k = 0; k < morsi; k++) {                                // morsi di fogliame da un lato
    const y0 = Math.round(top + (rnd() + 0.5) * (A - top - 28)), hh = Math.round(13 + (rnd() + 0.5) * 60), prof = Math.round(20 + (rnd() + 0.5) * 100);
    const sin = rnd() < 0, xa = sin ? 0 : Math.round(160 + yaw + 20 - prof), xb = sin ? Math.round(160 + yaw - 20 + prof) : L;
    for (let y = y0; y < Math.min(A, y0 + hh); y++) for (let x = Math.max(0, xa); x < Math.min(L, xb); x++) m[y * L + x] = 0;
  }
  const buchi = Math.floor((rnd() + 0.5) * 30);
  for (let k = 0; k < buchi; k++) { const x = Math.round(33 + (rnd() + 0.5) * 250), y = Math.round(top + (rnd() + 0.5) * (A - top - 8));
    for (let yy = y; yy < Math.min(A, y + 4); yy++) for (let xx = x; xx < x + 7; xx++) m[yy * L + xx] = 0; }
  return m;
}
let n = 0, incroci = 0, minW = 1e9;
for (let i = 0; i < 600; i++) {
  const g = geometriaCorridoio(sporca(), L, A); if (!g) continue;
  const par = nastroBordi(g.bordi, g.rette, yref); if (!par) continue;
  n++;
  const ys = []; for (let k = 0; k < 60; k++) ys.push(g.yAlto + (g.yBasso + 2 - g.yAlto) * k / 59);
  const q = puntiNastro(par, g.rette, ys, yref);
  const w = Math.min(...q.xr.map((x, j) => x - q.xl[j])); minW = Math.min(minW, w); if (w <= 0) incroci++;
}
prova('600 maschere sporche: le curve non si incrociano mai', n > 250 && incroci === 0, `(${n} valutate, incroci ${incroci}, larghezza minima ${minW.toFixed(1)} px su 320)`);
console.log(falliti ? `\n${falliti} prove fallite` : '\ntutte le prove passano'); process.exit(falliti ? 1 : 0);
