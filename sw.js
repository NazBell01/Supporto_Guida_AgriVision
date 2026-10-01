// Service worker: dopo il primo caricamento l'app (pagina, script, runtime ONNX e modello)
// funziona senza rete, come in vigna dove il telefono non prende.
// Pagina e script: prima la rete, la cache solo se manca il segnale (così un aggiornamento
// arriva subito). Runtime e modello: prima la cache (sono grandi e cambiano di rado).
const VERSIONE = 'agrivision-v2';
const GUSCIO = ['./', 'index.html', 'style.css', 'manifest.webmanifest', 'icona-192.png',
                'js/app.js', 'js/geometria.js', 'js/generale.js', 'js/disegno.js', 'js/config.js',
                'vendor/ort/ort.min.js'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(VERSIONE).then(c => c.addAll(GUSCIO)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== VERSIONE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const r = e.request, u = new URL(r.url);
  if (r.method !== 'GET' || u.origin !== location.origin) return;
  const pesante = u.pathname.includes('/vendor/') || u.pathname.includes('/modelli/');
  e.respondWith((async () => {
    const cache = await caches.open(VERSIONE);
    if (pesante) {
      const hit = await cache.match(r, { ignoreSearch: false });
      if (hit) return hit;
      const net = await fetch(r);
      if (net.ok && net.status === 200) cache.put(r, net.clone());
      return net;
    }
    try {
      const net = await fetch(r);
      if (net.ok) cache.put(r, net.clone());
      return net;
    } catch {
      return (await cache.match(r, { ignoreSearch: true })) || Response.error();
    }
  })());
});
