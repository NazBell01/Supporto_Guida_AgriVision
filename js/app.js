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
import { analizza as analizzaGenerale, normalizza as normalizzaProfondita, PARAM as PARAM_GEN } from './generale.js';

const $ = id => document.getElementById(id);
const [L, A] = GUIDA_DIM;

// ----------------------------------------------------------- impostazioni ---
const memoria = {
  leggi() { try { return JSON.parse(localStorage.getItem('agrivision') || '{}'); } catch { return {}; } },
  scrivi(o) { try { localStorage.setItem('agrivision', JSON.stringify(o)); } catch { /* modalità privata */ } },
};
const prefs = { corridoioCm: CONFIG.corridoioCm, soglia: CONFIG.soglia, modo: 'vigna',
                altezzaM: null, inclinazione: null, hfov: PARAM_GEN.hfovGradi, ...memoria.leggi() };
// Un modello per modalità: vigna = segmentazione del corridoio, generale = profondità (MiDaS small).
const MODELLI = {
  vigna:    { url: CONFIG.modello,           nome: 'guida.onnx',      dim: [L, A] },
  generale: { url: CONFIG.modelloProfondita, nome: 'profondita.onnx', dim: [256, 256] },
};

// ------------------------------------------------------------------ stato ---
const S = {
  sessioni: {}, sess: null, nomeIn: null, nomeOut: null, provider: '?', modo: prefs.modo, precRotta: null,
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
const piccoloG = document.createElement('canvas'); piccoloG.width = 256; piccoloG.height = 256;
const pctxG = piccoloG.getContext('2d', { willReadFrequently: true });
const tensoreG = new Float32Array(3 * 256 * 256);
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

async function creaSessione(buf, dim) {
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
      const nomeIn = s.inputNames[0], x = new ort.Tensor('float32', new Float32Array(3 * dim[0] * dim[1]), [1, 3, dim[1], dim[0]]);
      await s.run({ [nomeIn]: x });
      const t = performance.now(); await s.run({ [nomeIn]: x });
      S.msInf = performance.now() - t; S.provider = ep[0];
      return s;
    } catch (e) { ultimo = e; console.warn('provider', ep, e); }
  }
  throw ultimo;
}

async function usaModello(buf, etichetta, modo) {
  const M = MODELLI[modo];
  $('modello-stato').textContent = 'Modello: inizializzazione…';
  const sess = await creaSessione(buf, M.dim);
  const nomeIn = sess.inputNames[0];
  const dims = sess.inputMetadata?.[nomeIn]?.shape;
  if (dims && dims.length === 4 && typeof dims[2] === 'number' && (dims[2] !== M.dim[1] || dims[3] !== M.dim[0])) {
    throw new Error(`il modello attende ${dims[3]}×${dims[2]}, questa modalità ${M.dim[0]}×${M.dim[1]}`);
  }
  S.sessioni[modo] = { sess, nomeIn, nomeOut: sess.outputNames[0], provider: S.provider, ms: S.msInf, etichetta };
  if (modo === S.modo) mostraModello();
}

function mostraModello() {
  const m = S.sessioni[S.modo];
  if (!m) return;
  S.sess = m.sess; S.nomeIn = m.nomeIn; S.nomeOut = m.nomeOut; S.provider = m.provider;
  $('modello-stato').textContent = `Modello pronto (${m.etichetta}) · ${m.provider} · ${m.ms.toFixed(0)} ms/fotogramma`;
  progresso(1); $('btn-avvia').disabled = false; $('carica-modello').hidden = true; errore('');
}

async function caricaModello(modo = S.modo) {
  const M = MODELLI[modo];
  $('btn-avvia').disabled = true; $('btn-avvia').textContent = 'Avvia videocamera';
  if (S.sessioni[modo]) return mostraModello();
  $('carica-modello').firstChild.textContent = `Scegli il file ${M.nome}`;
  try {
    await usaModello(await scaricaModello(M.url), M.nome, modo);
  } catch (e) {
    console.error(e);
    if (modo !== S.modo) return;
    progresso(0);
    $('modello-stato').textContent = `Modello non trovato: scegli il file ${M.nome}.`;
    $('carica-modello').hidden = false;
    errore(`Impossibile caricare ${M.url} (${e.message}). Vedi modelli/LEGGIMI.md.`);
  }
}
$('file-modello').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  errore('');
  try { await usaModello(new Uint8Array(await f.arrayBuffer()), f.name, S.modo); }
  catch (er) { errore(`File non valido: ${er.message}`); }
});

// ------------------------------------------------------------- modalità -----
function impostaModo(modo) {
  S.modo = prefs.modo = modo; memoria.scrivi(prefs);
  document.querySelectorAll('[data-modo]').forEach(el => (el.hidden = el.dataset.modo !== modo));
  for (const r of document.querySelectorAll('input[name=modo]')) r.checked = r.value === modo;
  S.sess = null; errore('');
  $('modello-stato').textContent = 'Modello: in caricamento…'; progresso(0);
  caricaModello(modo);
}
document.querySelectorAll('input[name=modo]').forEach(r => r.addEventListener('change', () => impostaModo(r.value)));
const numOrNull = v => { const x = parseFloat(v); return Number.isFinite(x) && x > 0 ? x : null; };
for (const [id, chiave] of [['in-altezza', 'altezzaM'], ['in-incl', 'inclinazione'], ['in-hfov', 'hfov']]) {
  $(id).value = prefs[chiave] ?? '';
  $(id).addEventListener('change', e => {
    prefs[chiave] = chiave === 'inclinazione' ? (e.target.value === '' ? null : parseFloat(e.target.value)) : numOrNull(e.target.value);
    memoria.scrivi(prefs);
  });
}

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
  S.righello = new Righello(); S.t0 = performance.now(); S.precRotta = null;
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
  if (S.modo === 'generale') { await passoGenerale(Wf, Hf, vw, vh); return diagnostica(vw, vh); }
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

  diagnostica(vw, vh);
}

function diagnostica(vw, vh) {
  const n = performance.now();
  if (S.ultimoFps) S.fps = S.fps ? S.fps * 0.9 + (1000 / (n - S.ultimoFps)) * 0.1 : 1000 / (n - S.ultimoFps);
  S.ultimoFps = n;
  const d = $('diag'); d.hidden = false;
  d.textContent = `${S.provider} · ${S.msInf.toFixed(0)} ms · ${S.fps.toFixed(0)} fps · ${vw}×${vh}`;
}

// ------------------------------------------------------- modalità generale --
const ostCanvas = document.createElement('canvas'); ostCanvas.width = 256; ostCanvas.height = 256;

async function passoGenerale(Wf, Hf, vw, vh) {
  // la rete di profondità vuole 256×256: il fotogramma si stira, le proporzioni non contano
  pctxG.imageSmoothingEnabled = true; pctxG.imageSmoothingQuality = 'medium';
  pctxG.drawImage(video, 0, 0, 256, 256);
  normalizza(pctxG.getImageData(0, 0, 256, 256).data, [256, 256], tensoreG);
  const t = performance.now();
  const out = await S.sess.run({ [S.nomeIn]: new ort.Tensor('float32', tensoreG, [1, 3, 256, 256]) });
  S.msInf = S.msInf ? S.msInf * 0.9 + (performance.now() - t) * 0.1 : performance.now() - t;

  const nd = normalizzaProfondita(out[S.nomeOut].data);
  const vfov = 2 * Math.atan(Math.tan(prefs.hfov * Math.PI / 360) * vh / vw) * 180 / Math.PI;
  const cal = (prefs.altezzaM && prefs.inclinazione !== null)
    ? { altezzaM: prefs.altezzaM, inclinazioneGradi: prefs.inclinazione, vfovGradi: vfov } : null;
  const r = nd.valida ? analizzaGenerale(nd.d, 256, 256, S.precRotta, cal, { ...PARAM_GEN, hfovGradi: prefs.hfov }) : null;

  // la direzione si liscia nel tempo (media esponenziale) per non far tremare la freccia
  let rotta = null;
  if (r && r.azione && r.azione !== 'FERMO') {
    rotta = S.precRotta === null ? r.rotta : S.precRotta + 0.4 * (r.rotta - S.precRotta);
    S.precRotta = rotta;
  } else S.precRotta = null;
  const stato = !r || !r.azione ? 'incerto' : r.azione === 'FERMO' ? 'assente' : 'ok';
  const ora = (performance.now() - S.t0) / 1000;
  S.storia.push({ t: ora, v: rotta, s: stato });
  while (S.storia.length && S.storia[0].t < ora - CONFIG.secondiTraccia) S.storia.shift();
  const libero = !r || !r.pianoOk ? null : (r.metri ? r.metri[r.migliorBin] : r.chiaroMigliore);   // senza terreno riconosciuto nessun valore
  S.registro.push({ t: +ora.toFixed(3), misura: rotta === null ? null : +rotta.toFixed(1), stato,
                    azione: r ? r.azione : null, libero: libero === null || !Number.isFinite(libero) ? null : +libero.toFixed(2) });

  // ostacoli come velo rosso (256×256, poi ingrandito dal canvas)
  const mc = ostCanvas.getContext('2d'), img = mc.createImageData(256, 256), d = img.data;
  if (r && r.pianoOk) for (let i = 0; i < r.ostacoli.length; i++) if (r.ostacoli[i]) { const j = i * 4; d[j] = 235; d[j + 1] = 70; d[j + 2] = 70; d[j + 3] = 105; }
  mc.putImageData(img, 0, 0);

  const k = Math.max(0.8, Math.min(1.5, Math.min(Wf, Hf) / 560));
  ctx.drawImage(video, 0, 0, Wf, Hf);
  const metri = !!(r && r.metri);
  const frasi = { sx: 'VIRA A SINISTRA', dx: 'VIRA A DESTRA', centro: 'VIA LIBERA', nd: 'TERRENO NON RICONOSCIUTO', fermo: 'OSTACOLO VICINO' };
  cruscotto.disegnaGenerale(Wf, Hf, k, { stato, r, ostCanvas, rotta, dati: {
    scarto: rotta, stato, unita: '°', storia: S.storia, ora, generale: true, frasi,
    fraseStato: { ok: 'PERCORSO CALCOLATO', incerto: 'TERRENO NON RICONOSCIUTO', assente: 'FERMARSI' }[stato],
    pannello2: { titolo: 'SPAZIO LIBERO',
      valore: libero === null || !Number.isFinite(libero) ? '--' : metri ? libero.toFixed(1) : String(Math.round(libero * 100)),
      unita: metri ? 'm' : '%', nota: metri ? 'camera calibrata' : 'relativo · non calibrato' },
  } });
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
    sessione: CONFIG.sessione, errori_misurati_su: CONFIG.sessione, unita: S.modo === 'generale' ? 'gradi' : unita,
    corridoio_cm: prefs.corridoioCm, soglia_sterzo_cm: prefs.soglia,
    sorgente: S.daFile ? 'video' : 'camera', provider: S.provider, modalita: S.modo,
    scarti: S.registro.filter(r => r.misura !== null).map(r => r.misura), registro: S.registro,
  };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(doc)], { type: 'application/json' }));
  a.download = `agrivision_${quando}__scarti.json`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});
document.addEventListener('visibilitychange', () => { if (!document.hidden && S.attivo) schermoAcceso(); });

// Funzionamento offline dopo il primo caricamento (richiede https o localhost).
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

impostaModo(prefs.modo === 'generale' ? 'generale' : 'vigna');
