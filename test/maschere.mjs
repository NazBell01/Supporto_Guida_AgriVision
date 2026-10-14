// Generatori di maschere sintetiche (320×192) per le prove delle linee guida.
export const L = 320, A = 192;
export const yref = 0.95 * A;
let seed = 5; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;   // [-0.5, 0.5)
const norm = () => (rnd() + rnd() + rnd() + rnd()) * 1.7;                                  // ~gaussiana, sigma ≈ 1


export function pulita(k0, rumore = 0) {
  const m = new Uint8Array(L * A);
  for (let y = 40; y < A; y++) {
    const t = (y - 40) / (A - 41), c = 160 + k0 * (y - yref) ** 2, h = 30 + 70 * t;
    const a = Math.max(0, Math.round(c - h + rumore * norm())), b = Math.min(L - 1, Math.round(c + h + rumore * norm()));
    for (let x = a; x <= b; x++) m[y * L + x] = 1;
  }
  return m;
}

export function sporca() {
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
