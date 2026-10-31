import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import AppShell from "@/components/AppShell";
import Auth from "@/pages/Auth";
import Console from "@/pages/Console";
import Dashboard from "@/pages/Dashboard";
import Landing from "@/pages/Landing";
import PatientPortal from "@/pages/PatientPortal";
import Settings from "@/pages/Settings";
import { landingFor, useSession } from "@/lib/session";
import "@/styles.css";

/**
 * Path-based routing, hand-rolled.
 *
 * A router dependency would be a larger bundle than the rest of this
 * frontend, and there is no server route to protect: nginx already
 * rewrites unknown paths to index.html. What the app does need is a
 * session guard, because a bookmarked /dashboard must not render before
 * the token has been read back out of storage.
 */
function useRoute() {
  const [path, setPath] = useState(() => window.location.pathname || "/");
  const [query, setQuery] = useState(() => window.location.search);

  useEffect(() => {
    const sync = () => {
      setPath(window.location.pathname || "/");
      setQuery(window.location.search);
    };
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { path, query };
}

function go(href: string) {
  window.history.pushState({}, "", href);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

/** Sends an unauthenticated visitor to sign-in, remembering where they
 *  were headed so they land there afterwards. */
function Guard({
  role,
  children,
}: {
  role?: "doctor" | "patient";
  children: React.ReactNode;
}) {
  const session = useSession();

  useEffect(() => {
    if (!session) {
      go(`/signin?next=${encodeURIComponent(window.location.pathname)}`);
    } else if (role && session.account.role !== role) {
      // Signed in, but on someone else's area. Send them somewhere that
      // is actually theirs rather than showing an empty page.
      go(landingFor(session));
    }
  }, [session, role]);

  if (!session) return null;
  if (role && session.account.role !== role) return null;
  return <>{children}</>;
}

function NotFound() {
  return (
    <AppShell items={[{ href: "/", label: "Home" }]}>
      <div className="dash-empty card" style={{ marginTop: 60 }}>
        <div className="empty-mark" aria-hidden="true">?</div>
        <h2 style={{ fontSize: 18, fontWeight: 700 }}>Page not found</h2>
        <p>That address does not exist.</p>
        <a className="btn btn-ghost btn-sm" href="/">
          Go home
        </a>
      </div>
    </AppShell>
  );
}

function App() {
  const { path, query } = useRoute();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [path]);

  if (path === "/") return <Landing />;
  if (path === "/signin") return <Auth initialMode="signin" />;
  if (path === "/register") return <Auth initialMode="doctor" />;
  if (path === "/dashboard")
    return (
      <Guard role="doctor">
        <Dashboard />
      </Guard>
    );
  if (path === "/portal")
    return (
      <Guard role="patient">
        <PatientPortal />
      </Guard>
    );
  if (path === "/settings")
    return (
      <Guard>
        <Settings />
      </Guard>
    );
  if (path.startsWith("/app"))
    return (
      <Guard role="doctor">
        <Console initialPatient={new URLSearchParams(query).get("patient")} />
      </Guard>
    );
  return <NotFound />;
}

/** Intercepts in-app links so navigation stays client-side. */
function useLinkInterception() {
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0) return;
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const anchor = (e.target as HTMLElement)?.closest?.("a");
      if (!anchor) return;
      const href = anchor.getAttribute("href");
      if (!href || href.startsWith("http") || anchor.target === "_blank") return;
      if (href.startsWith("#")) return;
      e.preventDefault();
      go(href);
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);
}

function Root() {
  useLinkInterception();
  return <App />;
}

/* ── PWA ─────────────────────────────────────────────────────── */

if ("serviceWorker" in navigator) {
  // Registered after load so it never competes with the first paint.
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js")
      // A failed registration is invisible by design: the app works
      // without it, and a console error would alarm without helping.
      .catch(() => {});
  });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
