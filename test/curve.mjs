// Prove dello stile «linee guida» (nastro: linea centrale che flette + semilarghezza in prospettiva):
//  1. fedeltà su sentieri puliti: ritrova la curvatura vera e i bordi cadono sulla maschera;
//  2. robustezza su maschere SPORCHE (morsi di fogliame, buchi, rumore): le due curve non si incrociano mai.
import { geometriaCorridoio, nastroBordi, puntiNastro, estremoLontano } from '../js/geometria.js';
import { L, A, yref, pulita, sporca } from './maschere.mjs';
let falliti = 0;
const prova = (n, ok, i = '') => { console.log(`${ok ? '✓' : '✗'} ${n} ${i}`); if (!ok) falliti++; };

// ---- 1. sentieri puliti ---------------------------------------------------------------------
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

// ---- 3. nessuno spigolo: la linea non «si impenna» in lontananza ---------------------------------
function svolta(P) {
  let mx = 0;
  for (let i = 1; i < P.length - 1; i++) {
    const a = Math.atan2(P[i][1] - P[i - 1][1], P[i][0] - P[i - 1][0]), b = Math.atan2(P[i + 1][1] - P[i][1], P[i + 1][0] - P[i][0]);
    let d = Math.abs(b - a) * 180 / Math.PI; if (d > 180) d = 360 - d; mx = Math.max(mx, d);
  }
  return mx;
}
let n3 = 0, spigoli = 0, somma = 0;
for (let i = 0; i < 600; i++) {
  const g = geometriaCorridoio(sporca(), L, A); if (!g) continue;
  const par = nastroBordi(g.bordi, g.rette, yref); if (!par) continue;
  const ya = estremoLontano(g.rette, yref, g.yAlto, g.yBasso), ys = [];       // lo stesso taglio che usa il disegno
  for (let k = 0; k < 44; k++) ys.push(g.yBasso + 6 + (ya - g.yBasso - 6) * (1 - Math.pow(1 - k / 43, 1.6)));
  const q = puntiNastro(par, g.rette, ys, yref);
  const mx = Math.max(svolta(ys.map((y, k) => [q.xl[k], y])), svolta(ys.map((y, k) => [q.xr[k], y])), svolta(ys.map((y, k) => [q.xm[k], y])));
  n3++; somma += mx; if (mx > 25) spigoli++;
}
prova('600 maschere sporche: nessuna linea con uno spigolo oltre 25°', n3 > 250 && spigoli === 0, `(${n3} valutate, con spigolo ${spigoli}, svolta massima media ${(somma / n3).toFixed(1)}°)`);
console.log(falliti ? `\n${falliti} prove fallite` : '\ntutte le prove passano'); process.exit(falliti ? 1 : 0);
