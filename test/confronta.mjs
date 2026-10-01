// Confronta il porting JS con i risultati del notebook (riferimento.json).
import { readFileSync } from 'node:fs';
import { scartoLaterale, geometriaCorridoio, Righello } from '../js/geometria.js';

const casi = JSON.parse(readFileSync(new URL('./riferimento.json', import.meta.url)));
let falliti = 0;
const vicini = (a, b, tol = 1e-6) =>
  (a === null && b === null) || (a !== null && b !== null && Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)));

for (const c of casi) {
  const buf = Buffer.from(c.bits, 'base64');
  const m = new Uint8Array(c.L * c.A);
  for (let i = 0; i < m.length; i++) m[i] = (buf[i >> 3] >> (7 - (i & 7))) & 1;
  const sc = scartoLaterale(m, c.L, c.A);
  const g = geometriaCorridoio(m, c.L, c.A);
  const prove = [
    ['scarto',    sc,                    c.scarto],
    ['centro',    g ? g.centro : null,    c.centro],
    ['larghezza', g ? g.larghezza : null, c.larghezza],
  ];
  if (c.rette && g) {
    prove.push(['pl0', g.rette[0][0], c.rette[0][0]], ['pl1', g.rette[0][1], c.rette[0][1]],
               ['pr0', g.rette[1][0], c.rette[1][0]], ['pr1', g.rette[1][1], c.rette[1][1]]);
  } else if (!!c.rette !== !!g) { prove.push(['rette presenti', g ? 1 : 0, c.rette ? 1 : 0]); }
  const male = prove.filter(([, a, b]) => !vicini(a, b, 1e-4));
  console.log(`${male.length ? '✗' : '✓'} ${c.nome.padEnd(12)} scarto=${sc === null ? '--' : sc.toFixed(3)} larg=${g ? g.larghezza.toFixed(2) : '--'}`);
  for (const [n, a, b] of male) { console.log(`    ${n}: js=${a} python=${b}`); falliti++; }
}
// Righello: credibilità
const r = new Righello();
for (let i = 0; i < 10; i++) r.aggiorna(100);
const ok = r.credibile(120) && !r.credibile(40) && !r.credibile(250) && !r.credibile(null);
console.log(`${ok ? '✓' : '✗'} righello`); if (!ok) falliti++;
console.log(falliti ? `\n${falliti} differenze` : '\ntutto coincide con il notebook');
process.exit(falliti ? 1 : 0);
