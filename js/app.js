// AgriVision · simulatore in tempo reale nel browser del telefono.
// Videocamera -> lettera di casella 320x192 -> DeepLabV3 (ONNX, onnxruntime-web)
// -> scarto laterale e geometria del corridoio -> cruscotto su canvas.
// Tutto avviene sul dispositivo: nessun fotogramma lascia il telefono.

import { CONFIG } from './config.js';
import {
  GUIDA_DIM, rettangoloUtile, normalizza, maschera, analizzaMaschera,
  rettePerFotogramma, statoMisura, Righello, mediana,
} from './geometria.js';
import { Cruscotto } from './disegno.js';

const $ = id => document.getElementById(id);
const [L, A] = GUIDA_DIM;

// ----------------------------------------------------------- impostazioni ---
const memoria = {
  leggi() { try { return JSON.parse(localStorage.getItem('agrivision') || '{}'); } catch { return {}; } },
  scrivi(o) { try { localStorage.setItem('agrivision', JSON.stringify(o)); } catch { /* modalità privata */ } },
};
const prefs = { corridoioCm: CONFIG.corridoioCm, soglia: CONFIG.soglia, ...memoria.leggi() };

// ------------------------------------------------------------------ stato ---
const S = {
  sess: null, nomeIn: null, nomeOut: null, provider: '?',
  stream: null, facing: 'environment', daFile: false,
  attivo: false, pausa: false, occupato: false,
  righello: new Righello(), storia: [], registro: [],
  largh: [], larghLunga: [], t0: 0, msInf: 0, fps: 0, ultimoFps: 0,
  vw: 0, vh: 0, wakeLock: null,
};

const video = $('video'), schermo = $('schermo');
const ctx = schermo.getContext('2d');
const piccolo = document.createElement('canvas'); piccolo.width = L; piccolo.height = A;
const pctx = piccolo.getContext('2d', { willReadFrequently: true });
const tensore = new Float32Array(3 * L * A);
const cruscotto = new Cruscotto(ctx, { ...CONFIG, ...prefs });

// ------------------------------------------------------------------- UI -----
function errore(msg) { const e = $('errore'); e.hidden = !msg; e.textContent = msg || ''; }
function avviso(msg, ms = 6000) {
  const a = $('avviso'); a.textContent = msg; a.hidden = false;
  clearTimeout(avviso.t); avviso.t = setTimeout(() => (a.hidden = true), ms);
}
function numero(v) { return String(v).replace('.', ','); }
$('e-dentro').textContent = numero(CONFIG.erroreDentroCm);
$('e-fuori').textContent = numero(CONFIG.erroreFuoriCm);
$('sessione').textContent = CONFIG.sessione;
$('in-cm').value = prefs.corridoioCm ?? '';
$('in-soglia').value = prefs.soglia;
$('in-cm').addEventListener('change', e => {
  const v = parseFloat(e.target.value);
  prefs.corridoioCm = Number.isFinite(v) && v > 0 ? v : null; memoria.scrivi(prefs); applicaPrefs();
});
$('in-soglia').addEventListener('change', e => {
  const v = parseFloat(e.target.value); if (v > 0) { prefs.soglia = v; memoria.scrivi(prefs); applicaPrefs(); }
});
function applicaPrefs() { Object.assign(cruscotto.cfg, { corridoioCm: prefs.corridoioCm, soglia: prefs.soglia }); }

// Il logo è facoltativo: se logo.png non c'è, il cruscotto scrive il nome.
{
  const img = new Image();
  img.onload = () => { cruscotto.setLogo(img); $('logo').src = img.src; $('logo').hidden = false; };
  img.src = CONFIG.logo;
}

// ----------------------------------------------------------------- modello --
function progresso(f) { $('barra').firstElementChild.style.width = `${Math.round(f * 100)}%`; }

async function scaricaModello(url) {
  const r = await fetch(`${url}?v=${CONFIG.sessione}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const tot = +r.headers.get('content-length') || 0;
  const rd = r.body.getReader(), pezzi = []; let n = 0;
  for (;;) {
    const { done, value } = await rd.read(); if (done) break;
    pezzi.push(value); n += value.length;
    if (tot) { progresso(n / tot); $('modello-stato').textContent = `Modello: ${(n / 1e6).toFixed(1)} / ${(tot / 1e6).toFixed(1)} MB`; }
  }
  const buf = new Uint8Array(n); let o = 0; for (const p of pezzi) { buf.set(p, o); o += p.length; }
  return buf;
}

async function creaSessione(buf) {
  ort.env.wasm.wasmPaths = new URL('vendor/ort/', document.baseURI).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  // ORT toglie da solo un provider non disponibile senza dirlo: si controlla qui, così
  // l'etichetta mostrata dice quello che gira davvero.
  let gpu = false;
  try { gpu = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { /* nessuna GPU */ }
  const prove = (gpu ? [['webgpu', 'wasm']] : []).concat([['wasm']]);
  let ultimo;
  for (const ep of prove) {
    try {
      const s = await ort.InferenceSession.create(buf, { executionProviders: ep, graphOptimizationLevel: 'all' });
      // Riscaldamento: verifica che il provider funzioni davvero e misura il tempo.
      const nomeIn = s.inputNames[0], x = new ort.Tensor('float32', new Float32Array(3 * L * A), [1, 3, A, L]);
      await s.run({ [nomeIn]: x });
      const t = performance.now(); await s.run({ [nomeIn]: x });
      S.msInf = performance.now() - t; S.provider = ep[0];
      return s;
    } catch (e) { ultimo = e; console.warn('provider', ep, e); }
  }
  throw ultimo;
}

async function usaModello(buf, etichetta) {
  $('modello-stato').textContent = 'Modello: inizializzazione…';
  S.sess = await creaSessione(buf);
  S.nomeIn = S.sess.inputNames[0]; S.nomeOut = S.sess.outputNames[0];
  const dims = S.sess.inputMetadata?.[S.nomeIn]?.shape;
  if (dims && dims.length === 4 && (dims[2] !== A || dims[3] !== L) && typeof dims[2] === 'number') {
    throw new Error(`il modello attende ${dims[3]}×${dims[2]}, il simulatore ${L}×${A}`);
  }
  $('modello-stato').textContent = `Modello pronto (${etichetta}) · ${S.provider} · ${S.msInf.toFixed(0)} ms/fotogramma`;
  progresso(1); $('btn-avvia').disabled = false; $('carica-modello').hidden = true;
}

async function caricaModello() {
  try {
    await usaModello(await scaricaModello(CONFIG.modello), 'guida.onnx');
  } catch (e) {
    console.error(e);
    progresso(0);
    $('modello-stato').textContent = 'Modello non trovato: scegli il file guida.onnx esportato dal notebook.';
    $('carica-modello').hidden = false;
    errore(`Impossibile caricare ${CONFIG.modello} (${e.message}). Vedi modelli/LEGGIMI.md.`);
  }
}
$('file-modello').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  errore('');
  try { await usaModello(new Uint8Array(await f.arrayBuffer()), f.name); }
  catch (er) { errore(`File non valido: ${er.message}`); }
});

// ------------------------------------------------------------ sorgente video --
async function avviaCamera() {
  errore('');
  if (!navigator.mediaDevices?.getUserMedia) {
    return errore('Questo browser non consente l\'accesso alla videocamera (serve una pagina https).');
  }
  try {
    fermaStream();
    S.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: S.facing }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    S.daFile = false; video.removeAttribute('src'); video.loop = false; video.srcObject = S.stream;
    await video.play();
    partenza();
  } catch (e) {
    const m = { NotAllowedError: 'Permesso negato: consenti l\'uso della videocamera per questo sito e riprova.',
                NotFoundError: 'Nessuna videocamera trovata.', NotReadableError: 'La videocamera è in uso da un\'altra app.',
                OverconstrainedError: 'La videocamera richiesta non è disponibile.' }[e.name];
    errore(m || `Videocamera non disponibile: ${e.message}`);
  }
}

async function avviaFile(file) {
  errore(''); fermaStream();
  S.daFile = true; video.srcObject = null; video.src = URL.createObjectURL(file); video.loop = true; video.muted = true;
  try { await video.play(); partenza(); } catch (e) { errore(`Video non riproducibile: ${e.message}`); }
}

function fermaStream() { S.stream?.getTracks().forEach(t => t.stop()); S.stream = null; }

async function schermoAcceso() {
  try { S.wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* facoltativo */ }
}

function partenza() {
  S.attivo = true; S.pausa = false; S.storia = []; S.registro = []; S.largh = []; S.larghLunga = [];
  S.righello = new Righello(); S.t0 = performance.now();
  $('avvio').hidden = true; $('comandi').hidden = false; $('b-cam').hidden = S.daFile;
  schermoAcceso(); $('b-pausa').textContent = '❚❚';
  setTimeout(() => {
    if (S.vh > S.vw) avviso('Per la massima precisione tieni il telefono in orizzontale: il modello lavora su 320×192.');
  }, 800);
  requestAnimationFrame(ciclo);
}

function chiudi() {
  S.attivo = false; fermaStream(); video.pause(); video.removeAttribute('src'); video.srcObject = null;
  try { S.wakeLock?.release(); } catch { /* */ }
  $('avvio').hidden = false; $('comandi').hidden = true; $('diag').hidden = true;
  if (document.fullscreenElement) document.exitFullscreen?.();
}

// ------------------------------------------------------------------- ciclo --
function dimensionaCanvas(vw, vh) {
  // il fotogramma sta intero nella finestra (niente ritaglio): le coordinate del
  // canvas sono quelle del fotogramma, come nel notebook
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const s = Math.min(innerWidth / vw, innerHeight / vh);
  const w = Math.round(vw * s), h = Math.round(vh * s);
  if (schermo.dataset.k !== `${w}x${h}x${dpr}`) {
    schermo.dataset.k = `${w}x${h}x${dpr}`;
    schermo.style.width = `${w}px`; schermo.style.height = `${h}px`;
    schermo.width = Math.round(w * dpr); schermo.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

const maskCanvas = document.createElement('canvas');

async function ciclo() {
  if (!S.attivo) return;
  requestAnimationFrame(ciclo);
  if (S.pausa || S.occupato || video.readyState < 2 || !video.videoWidth) return;
  S.occupato = true;
  try { await passo(); } catch (e) { console.error(e); avviso(`Errore: ${e.message}`); S.pausa = true; }
  finally { S.occupato = false; }
}

async function passo() {
  const vw = video.videoWidth, vh = video.videoHeight; S.vw = vw; S.vh = vh;
  const { w: Wf, h: Hf } = dimensionaCanvas(vw, vh);
  const util = rettangoloUtile(vw, vh), { ox, oy, nw, nh } = util;

  // 1. lettera di casella e normalizzazione ImageNet
  pctx.fillStyle = '#000'; pctx.fillRect(0, 0, L, A);
  pctx.imageSmoothingEnabled = true; pctx.imageSmoothingQuality = 'medium';
  pctx.drawImage(video, ox, oy, nw, nh);
  normalizza(pctx.getImageData(0, 0, L, A).data, GUIDA_DIM, tensore);

  // 2. rete di guida
  const t = performance.now();
  const out = await S.sess.run({ [S.nomeIn]: new ort.Tensor('float32', tensore, [1, 3, A, L]) });
  const ms = performance.now() - t;
  S.msInf = S.msInf ? S.msInf * 0.9 + ms * 0.1 : ms;
  const m = maschera(out[S.nomeOut].data, GUIDA_DIM);

  // 3. scarto, geometria, righello
  const an = analizzaMaschera(m, S.righello, prefs.corridoioCm);
  const cm = !!prefs.corridoioCm;
  const misura = cm ? an.cm : an.sc;
  const stato = statoMisura(misura, true, an.affidabile);
  const ora = (performance.now() - S.t0) / 1000;
  S.storia.push({ t: ora, v: misura, s: stato });
  while (S.storia.length && S.storia[0].t < ora - CONFIG.secondiTraccia) S.storia.shift();
  S.registro.push({ t: +ora.toFixed(3), misura: misura === null ? null : +misura.toFixed(2), stato });

  // 4. disegno
  const k = Math.max(0.8, Math.min(1.5, Math.min(Wf, Hf) / 560));
  ctx.drawImage(video, 0, 0, Wf, Hf);
  const syf = Hf / nh;
  let geoF = null, yA = 0, yB = 0;
  if (an.geo) {
    geoF = rettePerFotogramma(an.geo.rette, util, Wf, Hf);
    yA = (an.geo.yAlto - oy) * syf; yB = (an.geo.yBasso + 1 - oy) * syf;
  }
  aggiornaMaschera(m, util);
  cruscotto.disegna(Wf, Hf, k, {
    stato, geoF, yAltoF: yA, yBassoF: yB, maskCanvas, areaMask: { x: 0, y: 0, w: Wf, h: Hf },
    dati: z => {
      // stima della larghezza: rapporto fra la larghezza di ora e quella tipica della
      // corsa (NON è una misura diretta: assume camera ad altezza e inclinazione costanti)
      let stima = null;
      if (z && z.largh) {
        S.largh.push(z.largh); if (S.largh.length > 45) S.largh.shift();
        S.larghLunga.push(z.largh); if (S.larghLunga.length > 9000) S.larghLunga.shift();
        if (cm && S.larghLunga.length >= 30) {
          stima = prefs.corridoioCm * mediana(S.largh.slice(-15)) / Math.max(mediana(S.larghLunga), 1e-6);
        }
      }
      return { scarto: misura, stato, unita: cm ? 'cm' : 'px', stimaCm: stima, storia: S.storia, ora };
    },
  });

  // FPS e diagnostica
  const n = performance.now();
  if (S.ultimoFps) S.fps = S.fps ? S.fps * 0.9 + (1000 / (n - S.ultimoFps)) * 0.1 : 1000 / (n - S.ultimoFps);
  S.ultimoFps = n;
  const d = $('diag'); d.hidden = false;
  d.textContent = `${S.provider} · ${S.msInf.toFixed(0)} ms · ${S.fps.toFixed(0)} fps · ${vw}×${vh}`;
}

// Maschera grezza come velo blu del marchio (alfa 0.14), ritagliata sull'area utile.
function aggiornaMaschera(m, { ox, oy, nw, nh }) {
  if (maskCanvas.width !== nw || maskCanvas.height !== nh) { maskCanvas.width = nw; maskCanvas.height = nh; }
  const mc = maskCanvas.getContext('2d'), img = mc.createImageData(nw, nh), d = img.data;
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
    if (m[(y + oy) * L + x + ox]) { const j = (y * nw + x) * 4; d[j] = 11; d[j + 1] = 64; d[j + 2] = 128; d[j + 3] = 36; }
  }
  mc.putImageData(img, 0, 0);
}

// --------------------------------------------------------------- comandi ----
$('btn-avvia').addEventListener('click', avviaCamera);
$('file-video').addEventListener('change', e => { const f = e.target.files[0]; if (f) avviaFile(f); e.target.value = ''; });
$('b-stop').addEventListener('click', chiudi);
$('b-pausa').addEventListener('click', () => {
  S.pausa = !S.pausa; $('b-pausa').textContent = S.pausa ? '▶' : '❚❚'; if (S.pausa) video.pause(); else video.play();
});
$('b-cam').addEventListener('click', () => { S.facing = S.facing === 'environment' ? 'user' : 'environment'; avviaCamera(); });
$('b-full').addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.().catch(() => {});
});
$('b-log').addEventListener('click', () => {
  const quando = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const unita = prefs.corridoioCm ? 'cm' : 'px';
  const doc = {
    sessione: CONFIG.sessione, errori_misurati_su: CONFIG.sessione, unita,
    corridoio_cm: prefs.corridoioCm, soglia_sterzo_cm: prefs.soglia,
    sorgente: S.daFile ? 'video' : 'camera', provider: S.provider,
    scarti: S.registro.filter(r => r.misura !== null).map(r => r.misura), registro: S.registro,
  };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(doc)], { type: 'application/json' }));
  a.download = `agrivision_${quando}__scarti.json`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});
document.addEventListener('visibilitychange', () => { if (!document.hidden && S.attivo) schermoAcceso(); });

// Funzionamento offline dopo il primo caricamento (richiede https o localhost).
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

caricaModello();
