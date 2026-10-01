const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const S = process.argv[2];
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${S}/prova.y4m`] });
  const ctx = await b.newContext({ viewport: { width: 844, height: 390 }, deviceScaleFactor: 2, permissions: ['camera'] });
  const p = await ctx.newPage();
  p.on('pageerror', e => console.log('PAGEERROR', e.message));
  await p.goto('http://localhost:8099/');
  await p.waitForSelector('#carica-modello:not([hidden])', { timeout: 20000 });
  await p.setInputFiles('#file-modello', `${S}/guida_finto.onnx`);
  await p.waitForFunction(() => !document.getElementById('btn-avvia').disabled, null, { timeout: 30000 });
  await p.click('#btn-avvia');
  for (const t of [1500, 2500, 2500]) {
    await p.waitForTimeout(t);
    console.log(await p.textContent('#diag'), '|', (await p.textContent('#avviso')) || '');
    await p.screenshot({ path: `${S}/cam_${t}_${Date.now() % 1000}.png` });
  }
  // registro
  const [dl] = await Promise.all([p.waitForEvent('download'), p.click('#b-log')]);
  const j = JSON.parse(require('fs').readFileSync(await dl.path()));
  const v = j.scarti; console.log('registro:', j.registro.length, 'fotogrammi, scarti min/max cm', Math.min(...v).toFixed(1), Math.max(...v).toFixed(1), j.unita);
  await b.close();
})();
