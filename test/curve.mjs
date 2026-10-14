// Prove della curvatura dei bordi (stile «linee guida»).
import { geometriaCorridoio, curveBordi, limitaCurvatura } from '../js/geometria.js';
const L = 320, A = 192;
function maschera(k0l, k0r, rumore = 0, seed = 5) {
  let s = seed; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647 - 0.5;
  const m = new Uint8Array(L * A), yref = 0.95 * A;
  for (let y = 40; y < A; y++) {
    const t = (y - 40) / (A - 41);
    const q = (y - yref) ** 2;
    let a = 140 - 70 * t + k0l * q + rumore * rnd() * 2, b = 180 + 70 * t + k0r * q + rumore * rnd() * 2;
    a = Math.max(0, Math.round(a)); b = Math.min(L - 1, Math.round(b));
    for (let x = a; x <= b; x++) m[y * L + x] = 1;
  }
  return m;
}
let falliti = 0;
const prova = (n, ok, i = '') => { console.log(`${ok ? '✓' : '✗'} ${n} ${i}`); if (!ok) falliti++; };
const yref = 0.95 * A;
const stima = (k0l, k0r, rum) => { const g = geometriaCorridoio(maschera(k0l, k0r, rum), L, A); return curveBordi(g.bordi, g.rette, yref).map(c => c.c2); };

let [kl, kr] = stima(0, 0);
prova('corridoio dritto → curvatura ~0', Math.abs(kl) < 2e-4 && Math.abs(kr) < 2e-4, `(${kl.toExponential(1)}, ${kr.toExponential(1)})`);
[kl, kr] = stima(0.004, 0.004);
prova('sentiero che piega da un lato → curvatura concorde e vicina al vero (±25%)', Math.abs(kl - 0.004) < 0.001 && Math.abs(kr - 0.004) < 0.001, `(${kl.toFixed(4)}, ${kr.toFixed(4)}; vera 0.0040)`);
[kl, kr] = stima(-0.004, -0.004);
prova('piega dall’altro lato → segno opposto, vicina al vero', Math.abs(kl + 0.004) < 0.001 && Math.abs(kr + 0.004) < 0.001, `(${kl.toFixed(4)}, ${kr.toFixed(4)})`);
[kl, kr] = stima(0.004, 0.004, 2.0);
prova('con rumore sulla maschera la curvatura resta riconoscibile', Math.abs(kl - 0.004) < 0.0015 && Math.abs(kr - 0.004) < 0.0015, `(${kl.toFixed(4)}, ${kr.toFixed(4)})`);
const g = geometriaCorridoio(maschera(0.004, 0.004), L, A);
const lim = limitaCurvatura(1.0, 150, g.larghezza);
prova('curvatura assurda → limitata', lim > 0 && lim < 1e-2, `(${lim.toExponential(2)})`);
console.log(falliti ? `\n${falliti} prove fallite` : '\ntutte le prove passano'); process.exit(falliti ? 1 : 0);
