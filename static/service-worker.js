const CACHE_NAME = "cropyield-v1";
const APP_SHELL = [
  "/",
  "/static/styles.css",
  "/static/app.js",
  "/static/manifest.json",
  "/static/icons/icon-192.svg",
  "/static/icons/icon-512.svg",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
        )
      )
  );
  self.clients.claim();
});

// Simple network-first for API requests, cache-first for others
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // API requests: try network, fallback to cache
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(event.request)
        .then((resp) => {
          // clone + cache
          const copy = resp.clone();
          caches
            .open(CACHE_NAME)
            .then((cache) => cache.put(event.request, copy));
          return resp;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // For navigation and other assets: cache-first
  event.respondWith(
    caches
      .match(event.request)
      .then(
        (cached) =>
          cached || fetch(event.request).catch(() => caches.match("/"))
      )
  );
});
