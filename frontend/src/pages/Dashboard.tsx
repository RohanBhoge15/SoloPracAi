import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import AppShell from "@/components/AppShell";
import PrescriptionComposer from "@/components/PrescriptionComposer";
import { api, ApiError, type Dashboard as DashboardData, type Patient } from "@/lib/api";
import { useSession } from "@/lib/session";
import "@/pages/dashboard.css";

const NAV = [
  { href: "/dashboard", label: "Overview" },
  { href: "/app", label: "Records" },
  { href: "/settings", label: "Settings" },
];

function greeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

function initials(name: string): string {
  return (
    name
      .split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((p) => p[0]?.toUpperCase() ?? "")
      .join("") || "?"
  );
}

/**
 * A patient's picture, or their initials.
 *
 * A patient who signs up with the email their doctor entered finds the
 * record automatically, and any picture they upload is mirrored onto
 * every patient row they own - so it reaches every doctor they have
 * seen, with nothing for the doctor to do. The `<img>` is pointed at a
 * presigned object-storage URL because an img tag cannot send an
 * Authorization header; if the fetch or the load fails, this falls back
 * to initials rather than leaving a hole.
 */
function PatientAvatar({ patient }: { patient: Patient }) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    if (!patient.photo_path) {
      setSrc(null);
      return;
    }
    void api.patientPhotoUrl(patient.id).then((url) => {
      if (live) setSrc(url);
    });
    return () => {
      live = false;
    };
  }, [patient.id, patient.photo_path]);

  if (!src) {
    return <span className="pavatar">{initials(patient.display_name)}</span>;
  }
  return (
    <img
      className="pavatar pavatar--img"
      src={src}
      alt=""
      onError={() => setSrc(null)}
    />
  );
}

function when(iso: string | null): string {
  if (!iso) return "no activity yet";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const mins = Math.floor((Date.now() - then) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

export default function Dashboard() {
  const session = useSession();
  const [data, setData] = useState<DashboardData | null>(null);
  const [search, setSearch] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [composing, setComposing] = useState(false);

  const load = useCallback(async (q: string) => {
    try {
      const result = await api.dashboard(q, 8);
      setData(result);
      setError(null);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Could not load the dashboard.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounced so typing does not fire a request per keystroke. 250 ms is
  // below the threshold where search feels laggy.
  useEffect(() => {
    const handle = window.setTimeout(() => void load(search), search ? 250 : 0);
    return () => window.clearTimeout(handle);
  }, [search, load]);

  const kpis = data?.kpis;

  const cards = useMemo(
    () =>
      kpis
        ? [
            {
              label: "Patients",
              value: kpis.patients,
              sub: kpis.patients_added_7d
                ? `+${kpis.patients_added_7d} this week`
                : "none added this week",
              tone: "accent",
            },
            {
              label: "Records",
              value: kpis.records,
              sub: `${kpis.records_added_today} today`,
              tone: "current",
            },
            {
              label: "Prescriptions",
              value: kpis.prescriptions,
              sub: `${kpis.documents_sent_today} documents sent today`,
              tone: "current",
            },
            {
              label: "Reports",
              value: kpis.reports,
              sub: `${kpis.certificates} certificates, ${kpis.invoices} invoices`,
              tone: "accent",
            },
          ]
        : [],
    [kpis],
  );

  return (
    <AppShell items={NAV}>
      <div className="dash-head">
        <div>
          <h1 className="dash-title">
            {greeting()}
            {session ? `, ${session.account.display_name.split(" ")[0]}` : ""}
          </h1>
          <p className="dash-sub">
            {data?.clinic_name ?? "Your practice at a glance"}
            {data?.specialty ? ` - ${data.specialty}` : ""}
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => setComposing(true)}
        >
          New prescription
        </button>
      </div>

      {error && <p className="dash-error">{error}</p>}

      {loading && !data && (
        <div className="kpi-grid">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="kpi card kpi-skeleton" />
          ))}
        </div>
      )}

      {kpis && (
        <div className="kpi-grid">
          {cards.map((card, i) => (
            <motion.div
              key={card.label}
              className={`kpi card tone-${card.tone}`}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: i * 0.06, ease: [0.22, 1, 0.36, 1] }}
            >
              <p className="kpi-label">{card.label}</p>
              <p className="kpi-value">{card.value.toLocaleString()}</p>
              <p className="kpi-sub">{card.sub}</p>
            </motion.div>
          ))}
        </div>
      )}

      {kpis && kpis.awaiting_review > 0 && (
        <a href="/app" className="review-warn">
          <span className="dot dot-pulse" />
          <strong>{kpis.awaiting_review}</strong> scan
          {kpis.awaiting_review === 1 ? "" : "s"} could not be read
          confidently and {kpis.awaiting_review === 1 ? "is" : "are"} waiting
          for review.
        </a>
      )}

      <section className="dash-section">
        <div className="dash-section-head">
          <h2 className="dash-h2">
            {search ? "Search results" : "Recent patients"}
          </h2>
          {data && (
            <span className="dash-count">
              showing {data.recent_patients.length} of {data.total_patients}
            </span>
          )}
        </div>

        <div className="dash-search">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <path d="m20 20-3.5-3.5" />
          </svg>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name, phone or email"
            aria-label="Search patients"
          />
          {search && (
            <button
              type="button"
              className="dash-clear"
              onClick={() => setSearch("")}
              aria-label="Clear search"
            >
              x
            </button>
          )}
        </div>

        {data && data.recent_patients.length === 0 && (
          <div className="dash-empty card">
            <p>
              {search
                ? "No patient matches that."
                : "No patients yet. Add one from the Records tab to get started."}
            </p>
          </div>
        )}

        <ul className="plist">
          {data?.recent_patients.map((p, i) => (
            <motion.li
              key={p.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.32, delay: i * 0.04, ease: [0.22, 1, 0.36, 1] }}
            >
              <a href={`/app?patient=${p.id}`} className="pcard card card-hover">
                <PatientAvatar patient={p} />
                <span className="pbody">
                  <span className="pname">{p.display_name}</span>
                  <span className="pmeta">
                    {[
                      p.age != null ? `${p.age}y` : null,
                      p.phone,
                      p.email,
                    ]
                      .filter(Boolean)
                      .join(" - ") || "no contact details"}
                  </span>
                </span>
                <span className="ptail">
                  <span className="pcount">{p.record_count} records</span>
                  <span className="pwhen">{when(p.last_activity)}</span>
                </span>
              </a>
            </motion.li>
          ))}
        </ul>
      </section>

      {composing && (
        <PrescriptionComposer
          onClose={() => setComposing(false)}
          onSent={() => {
            setComposing(false);
            void load(search);
          }}
        />
      )}
    </AppShell>
  );
}
