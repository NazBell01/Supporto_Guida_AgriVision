// Cruscotto AgriVision REV_R su canvas: trapezio, griglia, righello, console
// unica in basso, riga di stato. Niente mirino: solo l'asse della macchina.
// Coordinate in pixel CSS del canvas; `k` scala le misure del notebook (pensate
// per fotogrammi da ~1280x720) sullo schermo del telefono.

import { polyval } from './geometria.js';

// Palette del marchio (RGB). Blu pieno per i riempimenti, schiarito per linee e testo.
const BLU_MARCHIO = '#0B4080';
const C_CIANO   = '#50AAEB';
const C_ETICH   = '#E2E6E8';
const C_FONDO   = [26, 30, 34];
const C_ARANCIO = '#FAAA28';
const C_ROSSO   = '#EB4646';
const C_BIANCO  = '#F1F5F9';
const SCURO     = 'rgb(6,8,10)';

const COLORE_STATO = { ok: C_CIANO, incerto: C_ARANCIO, assente: C_ROSSO };
const FRASE_STATO  = { ok: 'MISURA VALIDA', incerto: 'MISURA INAFFIDABILE', assente: 'NESSUN CORRIDOIO' };

const ZONA_OPACITA = 0.34, ZONA_OPACITA_I = 0.18, MASCHERA_VELO = 0.14;
const GRIGLIA_TRASV = 20, GRIGLIA_LONG = 10;
const FONDO_RIQUADRO = 0.86;
const RIGA_RIGHELLO = 0.72;
const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

export class Cruscotto {
  /** @param {object} cfg {corridoioCm, soglia, erroreFuoriCm, secondiTraccia} */
  constructor(ctx, cfg) {
    this.ctx = ctx; this.cfg = cfg;
    this.logo = null;
    this.testataH = 0; this.piedeH = 0;
    this.k = 1;
  }
  setLogo(img) { this.logo = img; }
  K(v) { return Math.max(1, Math.round(v * this.k)); }

  // ------------------------------------------------------------- primitive --
  testo(s, x, y, scala, col, peso = 600) {
    const c = this.ctx, px = Math.max(8, scala * 34 * this.k);
    c.font = `${peso} ${px}px ${FONT}`; c.textBaseline = 'alphabetic';
    c.lineJoin = 'round'; c.lineWidth = Math.max(2, px * 0.28);
    c.strokeStyle = SCURO; c.strokeText(s, x, y);
    c.fillStyle = col; c.fillText(s, x, y);
  }
  larg(s, scala, peso = 600) {
    const c = this.ctx; c.font = `${peso} ${Math.max(8, scala * 34 * this.k)}px ${FONT}`;
    return c.measureText(s).width;
  }
  linea(x1, y1, x2, y2, col, sp) {
    const c = this.ctx; c.strokeStyle = col; c.lineWidth = sp;
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
  }
  mescola(x, y, w, h, rgb, a) {            // sfuma un rettangolo verso un colore
    const c = this.ctx; c.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
    c.fillRect(x, y, w, h);
  }
  squadre(col) {
    const { ctx: c } = this, m = this.K(14), lung = this.K(46);
    const { w, h } = this.dim; c.strokeStyle = col; c.lineWidth = this.K(2);
    for (const [x, y, dx, dy] of [[m, m, 1, 1], [w - m, m, -1, 1], [m, h - m, 1, -1], [w - m, h - m, -1, -1]]) {
      c.beginPath(); c.moveTo(x + dx * lung, y); c.lineTo(x, y); c.lineTo(x, y + dy * lung); c.stroke();
    }
  }
  riquadro(x, y, w, h, col) {
    const c = this.ctx;
    this.mescola(x, y, w, h, C_FONDO, FONDO_RIQUADRO);
    c.strokeStyle = col; c.lineWidth = this.K(1); c.strokeRect(x, y, w, h);
    const l = this.K(10); c.lineWidth = this.K(2);
    for (const [px, py, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
      c.beginPath(); c.moveTo(px + dx * l, py); c.lineTo(px, py); c.lineTo(px, py + dy * l); c.stroke();
    }
  }

  // --------------------------------------------------------------- zona -----
  // Disegna maschera grezza (velo), trapezio, griglia e righello.
  // Ritorna {cmPx, centroCorr, largh} come _disegna_zona, o null.
  zona(maskCanvas, geoF, yAltoF, yBassoF, stato, areaMask) {
    const c = this.ctx, { w: Wf, h: Hf } = this.dim, k = this;
    const col = COLORE_STATO[stato], incerto = stato !== 'ok';
    if (maskCanvas) {                       // velo della maschera grezza, sotto
      c.save(); c.globalAlpha = 1; c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high';
      c.drawImage(maskCanvas, areaMask.x, areaMask.y, areaMask.w, areaMask.h); c.restore();
    }
    if (!geoF) return null;
    const [pl, pr] = geoF;
    const yA = yAltoF, yB = yBassoF;
    const x = (p, y) => polyval(p, y);
    // trapezio pieno
    c.save();
    c.beginPath();
    c.moveTo(x(pl, yB), yB); c.lineTo(x(pl, yA), yA); c.lineTo(x(pr, yA), yA); c.lineTo(x(pr, yB), yB); c.closePath();
    c.fillStyle = BLU_MARCHIO; c.globalAlpha = incerto ? ZONA_OPACITA_I : ZONA_OPACITA; c.fill();
    // griglia, ritagliata nel trapezio
    c.clip(); c.globalAlpha = 0.55; c.strokeStyle = col; c.lineWidth = k.K(1);
    const bsA = x(pl, yA), bdA = x(pr, yA), bsB = x(pl, yB), bdB = x(pr, yB);
    for (let i = 1; i <= GRIGLIA_TRASV; i++) {
      if (incerto && i % 2) continue;
      const y = yB + (yA - yB) * Math.pow(i / (GRIGLIA_TRASV + 1), 1.45);
      c.beginPath(); c.moveTo(x(pl, y), y); c.lineTo(x(pr, y), y); c.stroke();
    }
    for (let i = 1; i < GRIGLIA_LONG; i++) {
      const t = i / GRIGLIA_LONG;
      c.beginPath(); c.moveTo(bsB + (bdB - bsB) * t, yB); c.lineTo(bsA + (bdA - bsA) * t, yA); c.stroke();
    }
    c.restore();
    // righello alla riga di misura
    const yref = Math.round(Math.min(Math.max(RIGA_RIGHELLO * Hf, this.testataH + k.K(50)),
                                     Hf - this.piedeH - k.K(40)));
    const bs = x(pl, yref), bd = x(pr, yref), largh = bd - bs;
    if (largh < 20) return null;
    const centro = (bs + bd) / 2;
    const cmPx = this.cfg.corridoioCm ? this.cfg.corridoioCm / largh : null;
    this.righello(col, yref, bs, bd, centro, cmPx);
    return { cmPx, centroCorr: centro, largh };
  }

  righello(col, yref, bs, bd, centroCorr, cmPx) {
    const c = this.ctx, { w: Wf, h: Hf } = this.dim, K = v => this.K(v);
    const centroImg = (Wf - 1) / 2;
    this.linea(Math.max(0, bs), yref, Math.min(Wf - 1, bd), yref, col, K(2));
    if (cmPx) {
      // banda di incertezza: l'errore del modello FUORI giornata
      const xa = Math.max(0, centroImg - this.cfg.erroreFuoriCm / cmPx);
      const xb = Math.min(Wf, centroImg + this.cfg.erroreFuoriCm / cmPx);
      if (xb > xa) this.mescola(xa, yref - K(30), xb - xa, K(60), [250, 170, 40], 0.14);
      for (let cm = -150; cm <= 150; cm += 25) {
        const x = centroCorr + cm / cmPx;
        if (!(bs < x && x < bd) || !(x > 0 && x < Wf)) continue;
        const grande = cm % 50 === 0;
        this.linea(x, yref, x, yref - K(grande ? 13 : 6), col, K(grande ? 2 : 1));
        if (grande) {
          const e = cm === 0 ? '0' : String(Math.abs(cm));
          this.testo(e, x - this.larg(e, 0.34) / 2, yref - K(19), 0.34, C_ETICH);
        }
      }
    }
    // asse della macchina e centro del corridoio, alla stessa riga
    this.linea(centroImg, yref - K(38), centroImg, yref + K(38), C_BIANCO, K(1));
    c.fillStyle = C_BIANCO; c.beginPath();
    c.moveTo(centroImg, yref + K(38) - K(7)); c.lineTo(centroImg - K(6), yref + K(38) + K(5));
    c.lineTo(centroImg + K(6), yref + K(38) + K(5)); c.closePath(); c.fill();
    this.linea(centroCorr, yref - K(32), centroCorr, yref + K(32), col, K(3));
    c.fillStyle = col; c.beginPath(); c.arc(centroCorr, yref, K(5), 0, Math.PI * 2); c.fill();
  }

  // ----------------------------------------------------------- console ------
  geomConsole() {
    const { w: Wf, h: Hf } = this.dim, K = v => this.K(v);
    const m = K(26), g = K(8), Wc = Wf - 2 * m, ph = K(90);
    if (Wf < Hf) {                          // verticale: due file e sotto la traccia
      const wp = Math.floor((Wc - g) / 2), phT = K(54), phC = ph + phT, yc0 = Hf - K(34) - phC;
      return { m, Wc, ph, phC, yc0, vert: true,
               s: [m, yc0, wp], r: [m + wp + g, yc0, Wc - wp - g], t: [m, yc0 + ph + g, Wc, phT] };
    }
    const wp = Math.floor(Wc * 0.30), wr = Math.floor(Wc * 0.26), yc0 = Hf - K(34) - ph;
    return { m, Wc, ph, phC: ph, yc0, vert: false,
             s: [m, yc0, wp], r: [m + wp + g, yc0, wr], t: [m + wp + g + wr + g, yc0, Wc - wp - wr - 2 * g, ph] };
  }

  // dati: {scarto, stato, unita, stimaCm, storia:[{t,v,s}], ora}
  console(dati) {
    const c = this.ctx, { w: Wf, h: Hf } = this.dim, K = v => this.K(v);
    const stato = dati.stato, col = COLORE_STATO[stato], G = this.geomConsole();
    const [xs, ys, ws] = G.s, [xr, yr] = G.r, [xt, yt, wt, ht] = G.t;
    this.riquadro(G.m, G.yc0, G.Wc, G.phC, col);
    const seps = G.vert ? [xs + ws + K(4)] : [xs + ws + K(4), xr + G.r[2] + K(4)];
    for (const xsep of seps) if (xsep < G.m + G.Wc - K(20)) this.linea(xsep, G.yc0 + K(8), xsep, G.yc0 + G.ph - K(8), C_ETICH, K(1));
    if (G.vert) this.linea(G.m + K(8), yt - K(4), G.m + G.Wc - K(8), yt - K(4), C_ETICH, K(1));

    // scarto laterale: numero, lato, barra con la zona DRITTO
    this.testo('SCARTO LATERALE', xs + K(10), ys + K(16), 0.34, C_ETICH);
    const v = dati.scarto, cm = dati.unita === 'cm';
    const vals = dati.storia.map(r => r.v).filter(x => x !== null);
    let fondo, soglia = null;
    if (cm) { fondo = Math.max(3 * this.cfg.soglia, 30); soglia = this.cfg.soglia; }
    else { const a = vals.map(Math.abs).sort((p, q) => p - q); fondo = a.length ? Math.max(30, a[Math.max(0, Math.ceil(a.length * 0.95) - 1)] * 1.3) : 30; }
    const testo = v !== null ? Math.abs(v).toFixed(0) : '--';
    this.testo(testo, xs + K(12), ys + K(46), 0.92, col, 800);
    this.testo(dati.unita, xs + K(18) + this.larg(testo, 0.92, 800), ys + K(46), 0.40, C_ETICH);
    let frase, colF;
    if (v === null) { frase = { incerto: 'MISURA INAFFIDABILE', assente: 'NESSUN CORRIDOIO' }[stato] || '--'; colF = stato === 'incerto' ? C_ARANCIO : C_ROSSO; }
    else if (soglia !== null && Math.abs(v) <= soglia) { frase = 'CORRIDOIO CENTRATO'; colF = col; }
    else { frase = v > 0 ? 'CORRIDOIO A DESTRA' : 'CORRIDOIO A SINISTRA'; colF = col; }
    this.testo(frase, xs + K(12), ys + K(64), 0.34, colF);
    const bx0 = xs + K(14), bx1 = xs + ws - K(10), by0 = ys + K(72), by1 = ys + K(82);
    const cxb = (bx0 + bx1) / 2, semi = (bx1 - bx0) / 2;
    if (soglia !== null) this.mescola(cxb - semi * soglia / fondo, by0, 2 * semi * soglia / fondo, by1 - by0, [226, 230, 232], 0.45);
    let mx = null;
    if (v !== null) {
      mx = cxb + Math.max(-1, Math.min(1, v / fondo)) * semi;
      c.fillStyle = Math.abs(v) > fondo ? C_ARANCIO : col;
      c.fillRect(Math.min(cxb, mx), by0, Math.abs(mx - cxb), by1 - by0);
    }
    c.strokeStyle = C_ETICH; c.lineWidth = K(1); c.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);
    this.linea(cxb, by0 - K(3), cxb, by1 + K(3), C_ETICH, K(1));
    if (mx !== null) { this.linea(mx, by0 - K(4), mx, by1 + K(4), 'rgb(10,8,6)', K(5)); this.linea(mx, by0 - K(4), mx, by1 + K(4), C_BIANCO, K(3)); }

    // corridoio: la sola stima di larghezza
    this.testo('CORRIDOIO', xr + K(10), yr + K(16), 0.34, C_ETICH);
    const hasScala = !!this.cfg.corridoioCm;
    const t2 = hasScala && dati.stimaCm != null ? '~' + dati.stimaCm.toFixed(0) : '--';
    this.testo(t2, xr + K(12), yr + K(46), 0.92, C_CIANO, 800);
    this.testo(hasScala ? 'cm  stima' : 'px', xr + K(18) + this.larg(t2, 0.92, 800), yr + K(46), 0.34, C_ETICH);
    if (!hasScala) this.testo('scala non misurata', xr + K(12), yr + K(64), 0.28, C_ARANCIO);

    // traccia dello scarto, asse del tempo fisso
    this.testo(`SCARTO  ${this.cfg.secondiTraccia}s`, xt + K(10), yt + K(16), 0.34, C_ETICH);
    this.traccia(xt + K(10), yt + K(22), wt - K(20), (G.vert ? ht : G.ph) - K(30), dati, col, cm ? 100 : Math.max(30, fondo), soglia);

    // riga di stato
    const t3 = `AUSILIO ALL'OPERATORE  //  ${FRASE_STATO[stato]}`;
    this.testo(t3, (Wf - this.larg(t3, 0.38)) / 2, Hf - K(16), 0.38, col);
    c.fillStyle = C_ARANCIO; c.beginPath(); c.arc(G.m + K(6), Hf - K(20), K(5), 0, Math.PI * 2); c.fill();
    this.testo('LIVE', G.m + K(18), Hf - K(16), 0.30, C_ETICH);
    this.testataH = K(74); this.piedeH = Hf - G.yc0;
  }

  traccia(x, y, w, h, dati, col, scala, soglia) {
    const c = this.ctx, K = v => this.K(v), my = y + h / 2, T = this.cfg.secondiTraccia;
    if (soglia) { const mezza = h / 2 * Math.min(1, soglia / scala); this.mescola(x, my - mezza, w, 2 * mezza, [226, 230, 232], 0.2); }
    this.linea(x, my, x + w, my, C_ETICH, K(1));
    const pts = dati.storia; if (pts.length < 2) return;
    const X = t => x + w - (dati.ora - t) / T * w;
    c.strokeStyle = col; c.lineWidth = K(1.5); c.beginPath();
    let giu = false;
    for (const r of pts) {
      if (r.v === null) { giu = false; continue; }
      const px = X(r.t), py = my - Math.max(-scala, Math.min(scala, r.v)) / scala * (h / 2);
      if (!giu) { c.moveTo(px, py); giu = true; } else c.lineTo(px, py);
    }
    c.stroke();
    for (const r of pts) if (r.s !== 'ok') this.linea(X(r.t), y + h, X(r.t), y + h - K(4), r.s === 'incerto' ? C_ARANCIO : C_ROSSO, K(1));
    const ult = pts[pts.length - 1];
    if (ult.v !== null) { c.fillStyle = col; c.beginPath(); c.arc(X(ult.t), my - Math.max(-scala, Math.min(scala, ult.v)) / scala * (h / 2), K(2.5), 0, Math.PI * 2); c.fill(); }
  }

  testata(stato) {
    const c = this.ctx, K = v => this.K(v), col = COLORE_STATO[stato], m = K(26);
    if (this.logo) {
      const lh = K(40), lw = this.logo.width / this.logo.height * lh, px = m + K(4), py = K(16), mm = K(10);
      c.fillStyle = 'rgba(248,248,248,0.9)'; c.fillRect(px - mm, py - mm, lw + 2 * mm, lh + 2 * mm);
      c.strokeStyle = col; c.lineWidth = K(1); c.strokeRect(px - mm, py - mm, lw + 2 * mm, lh + 2 * mm);
      c.drawImage(this.logo, px, py, lw, lh);
    } else {
      this.testo('KYDO ROBOTICS', m + K(4), K(38), 0.58, col, 800);
    }
  }

  // Un fotogramma completo. `v` = {geoF, yAltoF, yBassoF, maskCanvas, areaMask, stato, dati: z => ({...})}
  disegna(w, h, k, v) {
    this.dim = { w, h }; this.k = k;
    const col = COLORE_STATO[v.stato];
    this.testataH = this.K(74);
    this.piedeH = h - this.geomConsole().yc0;
    this.squadre(col);
    this.testata(v.stato);
    const z = this.zona(v.maskCanvas, v.geoF, v.yAltoF, v.yBassoF, v.stato, v.areaMask);
    // asse della macchina: unica riga verticale a metà fotogramma
    const c = this.ctx;
    this.linea((w - 1) / 2, this.K(96), (w - 1) / 2, h - this.K(40), 'rgb(18,14,10)', this.K(3));
    this.linea((w - 1) / 2, this.K(96), (w - 1) / 2, h - this.K(40), C_BIANCO, this.K(1));
    this.console(v.dati(z));
    return z;
  }
}
