import { useEffect, useRef, useState, type ReactNode } from "react";
import Avatar, { useOwnPhotoUrl } from "@/components/Avatar";
import NotificationBell from "./NotificationBell";
import { clearSession, landingFor, useSession } from "@/lib/session";
import "@/components/shell.css";

export interface NavItem {
  href: string;
  label: string;
}

/**
 * Chrome shared by the doctor and patient areas.
 *
 * The nav items differ by role, and they are passed in rather than
 * derived here, because what a doctor can reach and what a patient can
 * reach is a product decision, not a rendering detail.
 */
export default function AppShell({
  items,
  children,
}: {
  items: NavItem[];
  children: ReactNode;
}) {
  const session = useSession();
  const [menu, setMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const path = window.location.pathname;

  useEffect(() => {
    if (!menu) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenu(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [menu]);

  const account = session?.account;
  // The top bar shows the same picture as Settings. Fetched here because
  // the account's own photo needs an authenticated request first: an
  // `<img>` cannot send the Authorization header, so pointing the tag
  // straight at the API 401s and the avatar silently renders blank.
  const photoUrl = useOwnPhotoUrl(Boolean(account?.photo_path));

  function signOut() {
    clearSession();
    window.location.href = "/";
  }

  return (
    <div className="shell">
      <div className="aurora" />
      <div className="grid-overlay" />

      <header className="topbar">
        <div className="topbar-inner">
          <a href={landingFor(session)} className="brand">
            <span className="brand-mark">S</span>
            <span className="brand-name">SoloPrac</span>
          </a>

          <nav className="topnav">
            {items.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className={`topnav-link ${
                  path.startsWith(item.href) ? "on" : ""
                }`}
              >
                {item.label}
              </a>
            ))}
          </nav>

          <div className="topbar-right">
            <NotificationBell />

            <div className="usermenu" ref={menuRef}>
              <button
                type="button"
                className="avatar-btn"
                onClick={() => setMenu((v) => !v)}
                aria-label="Account menu"
                aria-expanded={menu}
              >
                <Avatar
                  name={account?.display_name}
                  photoUrl={photoUrl}
                  photoKey={account?.photo_version}
                  className="avatar-in-btn"
                />
              </button>

              {menu && (
                <div className="usermenu-panel card">
                  <div className="usermenu-head">
                    <p className="usermenu-name">{account?.display_name}</p>
                    <p className="usermenu-mail">{account?.email}</p>
                    <span className="badge badge-accent">
                      {account?.role === "doctor" ? "Doctor" : "Patient"}
                    </span>
                  </div>
                  <a href="/settings" className="usermenu-item">
                    Profile &amp; settings
                  </a>
                  <button
                    type="button"
                    className="usermenu-item danger"
                    onClick={signOut}
                  >
                    Sign out
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="topmain">{children}</main>
    </div>
  );
}
