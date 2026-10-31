/* ============================================================
   Session.

   The token is an HMAC-signed blob from the API, not a JWT: it
   carries no claims, so there is nothing in it to trust. Every
   authorisation decision is made server-side against the session.

   localStorage rather than a cookie because the API is
   token-header based, and because the app is same-origin behind
   nginx. The trade-off is that this is readable by any script on
   the origin; that is acceptable only while the origin serves
   nothing but this app, which is why there is no third-party
   script in the bundle.
   ============================================================ */

const TOKEN_KEY = "solum.token";
const ACCOUNT_KEY = "solum.account";

export interface Account {
  id: string;
  email: string;
  role: "doctor" | "patient";
  display_name: string;
  phone: string | null;
  photo_path: string | null;
  photo_version: number;
  created_at: string;
}

export interface Session {
  token: string;
  account: Account;
}

/* ── storage ───────────────────────────────────────────────── */

function safeGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    // Private browsing, or storage disabled. The session then lasts
    // for the life of the tab, which is a degradation rather than
    // a failure, so it must not throw during render.
    return null;
  }
}

function safeSet(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* see safeGet */
  }
}

function safeRemove(key: string) {
  try {
    window.localStorage.removeItem(key);
  } catch {
    /* see safeGet */
  }
}

export function loadSession(): Session | null {
  const token = safeGet(TOKEN_KEY);
  const raw = safeGet(ACCOUNT_KEY);
  if (!token || !raw) return null;
  try {
    return { token, account: JSON.parse(raw) as Account };
  } catch {
    // A corrupted entry is discarded rather than left to throw on
    // every render.
    clearSession();
    return null;
  }
}

export function saveSession(session: Session) {
  safeSet(TOKEN_KEY, session.token);
  safeSet(ACCOUNT_KEY, JSON.stringify(session.account));
}

export function updateAccount(account: Account) {
  safeSet(ACCOUNT_KEY, JSON.stringify(account));
}

export function clearSession() {
  safeRemove(TOKEN_KEY);
  safeRemove(ACCOUNT_KEY);
}

export function getToken(): string | null {
  return safeGet(TOKEN_KEY);
}

/* ── reactive session ──────────────────────────────────────── */

import { useSyncExternalStore } from "react";

let current: Session | null = loadSession();
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((fn) => fn());
}

export function setSession(session: Session | null) {
  current = session;
  if (session) saveSession(session);
  else clearSession();
  emit();
}

export function setAccount(account: Account) {
  if (!current) return;
  current = { ...current, account };
  updateAccount(account);
  emit();
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

function snapshot(): Session | null {
  return current;
}

/**
 * Subscribes to the session.
 *
 * `getServerSnapshot` returns the same value, so this renders the
 * cached account immediately on hydration and then re-renders if a
 * refresh from the API contradicts it. Gating a protected page on
 * the API round-trip instead would flash the sign-in screen at
 * every reload.
 */
export function useSession(): Session | null {
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

/* ── helpers ───────────────────────────────────────────────── */

export function isDoctor(session: Session | null): boolean {
  return session?.account.role === "doctor";
}

export function isPatient(session: Session | null): boolean {
  return session?.account.role === "patient";
}

/** Where to send someone after they sign in. */
export function landingFor(session: Session | null): string {
  if (!session) return "/signin";
  return session.account.role === "doctor" ? "/dashboard" : "/portal";
}
