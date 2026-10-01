// Prove della modalità generale su scene SINTETICHE di disparità (terreno piano + ostacoli).
import { analizza, normalizza } from '../js/generale.js';

const W = 256, H = 256;
function scena(ostacoli = [], rumore = 0.01, seed = 3) {
  let s = seed; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647 - 0.5;
  const d = new Float32Array(W * H), v0 = 90;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    d[y * W + x] = Math.max(0, (y - v0) / (H - v0)) * 0.9 + 0.05 + rumore * rnd() * 2;   // piano
  }
  for (const o of ostacoli) {                      // scatola: parte da rigaBase, sporge in su, disparità costante
    for (let y = o.y0; y <= o.y1; y++) for (let x = o.x0; x <= o.x1; x++) d[y * W + x] = Math.max(d[y * W + x], o.d);
  }
  return d;
}
let falliti = 0;
const prova = (nome, ok, info = '') => { console.log(`${ok ? '✓' : '✗'} ${nome} ${info}`); if (!ok) falliti++; };

// 1. terreno libero: dritto
let r = analizza(normalizza(scena()).d, W, H);
prova('terreno libero → DRITTO', r.pianoOk && r.azione === 'DRITTO', `(rotta ${r.rotta.toFixed(1)}°, libero ${r.chiaroMigliore.toFixed(2)})`);

// 2. ostacolo a destra davanti: va a sinistra o dritto, non a destra
r = analizza(normalizza(scena([{ x0: 150, x1: 250, y0: 100, y1: 225, d: 0.9 }])).d, W, H);
prova('ostacolo a destra → non vira a destra', r.rotta <= 0.5 && r.azione !== 'DESTRA', `(${r.azione}, ${r.rotta.toFixed(1)}°)`);

// 3. ostacolo a sinistra
r = analizza(normalizza(scena([{ x0: 5, x1: 110, y0: 100, y1: 225, d: 0.9 }])).d, W, H);
prova('ostacolo a sinistra → non vira a sinistra', r.rotta >= -0.5 && r.azione !== 'SINISTRA', `(${r.azione}, ${r.rotta.toFixed(1)}°)`);

// 4. ostacolo al centro: sterza da un lato
r = analizza(normalizza(scena([{ x0: 100, x1: 160, y0: 110, y1: 215, d: 0.85 }])).d, W, H);
prova('ostacolo al centro → sterza', r.azione === 'DESTRA' || r.azione === 'SINISTRA', `(${r.azione}, ${r.rotta.toFixed(1)}°)`);
prova('   … e il bin centrale risulta bloccato', r.bloccato[Math.floor(r.nB / 2)] === 1);

// 5. muro che chiude tutto davanti, vicino: fermo
r = analizza(normalizza(scena([{ x0: 0, x1: 255, y0: 100, y1: 225, d: 0.95 }])).d, W, H);
prova('muro vicino su tutto il campo → FERMO', r.azione === 'FERMO', `(${r.azione}, piano ${r.pianoOk})`);

// 6. immagine piatta (nessuna profondità): nessun piano, nessuna misura inventata
const piatto = new Float32Array(W * H).fill(0.5); for (let i = 0; i < piatto.length; i += 97) piatto[i] += 1e-4;
r = analizza(normalizza(piatto).d, W, H);
prova('immagine senza struttura → nessuna direzione inventata', !r.pianoOk && r.azione === null, `(dentro ${r.dentro.toFixed(2)})`);

// 7. calibrazione: metri crescenti con la riga più alta
const cal = { altezzaM: 0.6, inclinazioneGradi: 20, vfovGradi: 50 };
r = analizza(normalizza(scena([{ x0: 100, x1: 160, y0: 110, y1: 215, d: 0.85 }])).d, W, H, null, cal);
prova('con camera calibrata lo spazio è in metri', r.metri && r.metri[Math.floor(r.nB / 2)] < r.metri[2], `(centro ${r.metri[24].toFixed(1)} m, lato ${r.metri[2].toFixed(1)} m)`);

// 8. parete di sbieco: la disparità cambia di lato, non in verticale → NON è terreno
{
  const d = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) d[y * W + x] = 0.1 + 0.85 * (x / W) + 0.1 * (y / H);
  r = analizza(normalizza(d).d, W, H);
  prova('parete di sbieco → non scambiata per terreno', !r.pianoOk && r.azione === null, `(angolo ${r.angolo.toFixed(0)}°)`);
}
// 9. telefono inclinato di lato (roll ~12°): il terreno resta terreno
{
  const d = scena(), roll = Math.tan(12 * Math.PI / 180);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) d[y * W + x] += 0.9 / (H - 90) * roll * (x - W / 2);
  r = analizza(normalizza(d).d, W, H);
  prova('telefono inclinato di 12° → terreno riconosciuto', r.pianoOk, `(angolo ${r.angolo.toFixed(0)}°)`);
}

// 10-12. Profondità VERE di MiDaS small su foto reali (fixture: test/profondita_reali.json)
{
  const { readFileSync } = await import('node:fs');
  const reali = JSON.parse(readFileSync(new URL('./profondita_reali.json', import.meta.url)));
  const carica = n => { const b = Buffer.from(reali[n].q, 'base64'), f = new Float32Array(W * H), { lo, hi } = reali[n];
    for (let i = 0; i < f.length; i++) f[i] = lo + (b.readUInt16LE(i * 2) / 65535) * (hi - lo); return f; };
  r = analizza(normalizza(carica('left')).d, W, H);
  prova('foto reale: libri su un pavimento → ostacoli rilevati, direzione verso il libero',
        r.pianoOk && r.bloccato.filter(Boolean).length >= 10 && r.azione === 'DESTRA', `(${r.azione}, ${r.rotta.toFixed(0)}°, ${r.bloccato.filter(Boolean).length}/${r.nB} direzioni bloccate)`);
  r = analizza(normalizza(carica('building')).d, W, H);
  prova('foto reale: facciata di un palazzo → nessuna via libera inventata', r.azione === null, `(angolo ${r.angolo.toFixed(0)}°)`);
  r = analizza(normalizza(carica('aloeL')).d, W, H);
  prova('foto reale: primo piano di una pianta → nessuna via libera inventata', r.azione === null, `(angolo ${r.angolo.toFixed(0)}°)`);
}

console.log(falliti ? `\n${falliti} prove fallite` : '\ntutte le prove passano'); process.exit(falliti ? 1 : 0);
