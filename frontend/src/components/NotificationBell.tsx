import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { api, type Notification } from "@/lib/api";

/**
 * Notification bell.
 *
 * Polls rather than holding a socket open. The stream is Redis-backed on
 * the server, but a long-lived websocket per tab buys little for a badge
 * that a doctor glances at between patients, and a polling loop fails
 * quietly instead of half-working. 30 s is short enough that a
 * prescription issued from another tab shows up while it is still
 * relevant.
 */
const POLL_MS = 30_000;

function ago(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

export default function NotificationBell() {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<Notification[]>([]);
  const [unread, setUnread] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [list, counts] = await Promise.all([
        api.notifications(20),
        api.unreadCount(),
      ]);
      setItems(list);
      setUnread(counts.unread);
      setError(null);
    } catch (err) {
      // A failed poll is not worth interrupting anyone over, but the user
      // should know the count is stale rather than zero.
      setError(
        err instanceof Error && err.message.includes("Cannot reach")
          ? "Offline"
          : null,
      );
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      // Not while the panel is open: re-rendering the list out from
      // under a cursor that is about to click something is hostile.
      if (!open) void refresh();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh, open]);

  /* Refetch whenever the tab comes back to the foreground.

     The interval above is the steady state, but a backgrounded tab is the
     case that actually breaks on a phone: iOS Safari suspends timers
     entirely, and Chrome throttles a hidden tab to about one poll a minute.
     Someone who leaves the app to read a prescription and comes back would
     otherwise wait up to a minute, or -- if the tab was fully suspended --
     see nothing at all until they forced a manual reload.

     `visibilitychange` is the right signal rather than `focus`: it also
     fires when a phone app is switched away from and back, and it does not
     fire spuriously when focus moves between two windows.

     The panel stays closed on this path. Refreshing under an open dropdown
     is exactly the hostility the interval above avoids, and this is a
     background-to-foreground transition, so the panel is essentially never
     open. */
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible" && !open) void refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    // Covers the desktop case where the window is restored from minimise:
    // some browsers do not fire visibilitychange for that.
    window.addEventListener("pageshow", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("pageshow", onVisible);
    };
  }, [refresh, open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  async function markAll() {
    const ids = items.filter((n) => !n.read).map((n) => n.id);
    if (ids.length === 0) return;
    // Optimistic: the badge is a small thing and waiting a round-trip
    // makes the control feel broken.
    setUnread(0);
    setItems((prev) => prev.map((n) => ({ ...n, read: true })));
    try {
      await api.markRead(ids);
    } catch {
      void refresh();
    }
  }

  return (
    <div className="bell" ref={rootRef}>
      <button
        type="button"
        className={`bell-btn ${open ? "on" : ""}`}
        onClick={() => {
          setOpen((v) => !v);
          if (!open) void refresh();
        }}
        aria-label={`Notifications${unread ? `, ${unread} unread` : ""}`}
        aria-expanded={open}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.73 21a2 2 0 0 1-3.46 0" />
        </svg>
        {unread > 0 && (
          <span className="bell-badge">
            {unread > 9 ? "9+" : unread}
          </span>
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            className="bell-panel card"
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="bell-head">
              <strong>Notifications</strong>
              {unread > 0 && (
                <button type="button" onClick={markAll} className="bell-clear">
                  Mark all read
                </button>
              )}
            </div>

            {error && <p className="bell-empty bell-offline">{error}</p>}

            {items.length === 0 && !error && (
              <p className="bell-empty">Nothing yet.</p>
            )}

            <ul className="bell-list">
              {items.map((n) => (
                <li key={n.id} className={`bell-item ${n.read ? "" : "unread"}`}>
                  <span className={`bell-kind k-${n.kind}`} aria-hidden="true" />
                  <div className="bell-body">
                    <p className="bell-title">{n.title}</p>
                    {n.body && <p className="bell-text">{n.body}</p>}
                    <time className="bell-time">{ago(n.created_at)}</time>
                  </div>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
