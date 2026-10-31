/* ============================================================
   Service worker.

   Deliberately conservative. This is a clinical system, and the worst
   possible outcome for a cache is serving one patient's record to
   another person on a shared device. So:

     - nothing under /api is ever cached
     - the app shell is cached, but only as a fallback
     - the token lives in localStorage, not in a cache this file owns

   There is no offline write path. A prescription composed on a train
   with no signal would have nowhere to go, and a queue that silently
   drops items is worse than an honest "you appear to be offline".
   ============================================================ */

const VERSION = "soloprac-v1";
const SHELL = ["/", "/index.html", "/manifest.webmanifest", "/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(VERSION)
      // addAll is atomic: if any one file 404s the whole install fails,
      // which is the behaviour we want rather than a half-cached shell.
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;

  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Anything from the API is never served from cache. A cached clinical
  // response is a disclosure waiting to happen, and staleness in a
  // medical record is the exact failure this project exists to prevent.
  if (url.pathname.startsWith("/api/") || url.pathname === "/health") return;

  // Presigned document URLs point at object storage on another origin.
  // Leave them entirely alone.
  if (url.origin !== self.location.origin) return;

  // Navigations: try the network, fall back to the cached shell so a
  // dropped connection still shows the app rather than the browser's
  // dinosaur.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put("/index.html", copy));
          return response;
        })
        .catch(() => caches.match("/index.html").then((hit) => hit || Response.error())),
    );
    return;
  }

  // Static assets: cache first. Vite fingerprints these filenames, so a
  // changed file is a changed URL and there is no stale-hit risk.
  if (url.pathname.startsWith("/assets/")) {
    event.respondWith(
      caches.match(request).then(
        (hit) =>
          hit ||
          fetch(request).then((response) => {
            const copy = response.clone();
            caches.open(VERSION).then((cache) => cache.put(request, copy));
            return response;
          }),
      ),
    );
  }
});

/* ── push ────────────────────────────────────────────────────
   Unused for now. The notification surface in this build is the
   in-app bell, which polls. Wiring Web Push needs a VAPID key pair,
   a subscription store and a delivery path, and none of that should
   be half-built. When it arrives, this is where the handler goes. */
