import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import DNAHelix from "@/components/DNAHelix";
import { api, type Health } from "@/lib/api";
import "./landing.css";

/* ── shared motion ──────────────────────────────────────────── */

const rise = {
  hidden: { opacity: 0, y: 26 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.07 * i, duration: 0.62, ease: [0.22, 1, 0.36, 1] as const },
  }),
};

const viewport = { once: true, margin: "-80px" } as const;

/* ── nav ────────────────────────────────────────────────────── */

function Nav() {
  return (
    <nav className="nav">
      <div className="brand">
        <div className="brand-mark">S</div>
        <span>SoloPrac</span>
      </div>
      <div className="nav-links">
        <a className="nav-link" href="#problem">Problem</a>
        <a className="nav-link" href="#method">Method</a>
        <a className="nav-link" href="#evidence">Evidence</a>
        <a className="btn btn-primary btn-sm" href="/app">
          Login/Register
        </a>
      </div>
    </nav>
  );
}

/* ── hero ───────────────────────────────────────────────────── */

function Hero() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <section className="hero">
      <div className="container hero-grid">
        <motion.div
          className="hero-copy"
          initial="hidden"
          animate="show"
          variants={rise}
          custom={0}
        >
          <div className="eyebrow">
            <span className="dot dot-pulse" style={{ color: "var(--current)" }} />
            Validity-aware clinical retrieval
          </div>

          <h1>
            Clinical AI can quote a
            <br />
            <span className="grad-text">retracted lab result</span>
            <br />
            and be clinically wrong.
          </h1>

          <p className="hero-lead">
            Every clinical RAG system treats a patient record as a list of things
            that are <em>true</em>. None of them model that a record can stop
            being true. SoloPrac represents the record as three-valued —{" "}
            <span style={{ color: "var(--current)" }}>current</span>,{" "}
            <span style={{ color: "var(--superseded)" }}>superseded</span>,{" "}
            <span style={{ color: "var(--accent)" }}>as-of</span> — and refuses
            to assert a corrected value as fact.
          </p>

          <div className="hero-cta">
            <a className="btn btn-primary" href="/app">
              Try it on a live record →
            </a>
            <a className="btn btn-ghost" href="#method">
              How it works
            </a>
          </div>

          <div className="hero-trust">
            <div className="trust-item">
              <span className="dot" style={{ color: "var(--current)" }} />
              {health?.llm_configured ? (
                <>
                  LLM live · <span className="mono">{health.llm_model}</span>
                </>
              ) : (
                "Extractive mode (no LLM key)"
              )}
            </div>
            <div className="trust-item">
              <span className="dot" style={{ color: "var(--accent)" }} />
              Runs locally · DPDP-aligned
            </div>
            <div className="trust-item">
              <span className="dot" style={{ color: "var(--text-3)" }} />
              159 tests · 78% coverage
            </div>
          </div>
        </motion.div>

        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
        >
          <DNAHelix />
        </motion.div>
      </div>
    </section>
  );
}

/* ── problem ────────────────────────────────────────────────── */

function Problem() {
  return (
    <section className="section" id="problem">
      <div className="container">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={viewport}
          variants={rise}
          custom={0}
        >
          <div className="eyebrow">The failure mode</div>
          <h2 className="section-title">
            A correct citation is not a correct answer
          </h2>
          <p className="section-sub">
            A lab reports a potassium of 2.1 mEq/L. Two days later it is amended
            to 4.8. Standard retrieval has no way to know the first value stopped
            being true, so it quotes it — faithfully, with a valid citation — and
            is wrong. Nothing flags it. There is no low-confidence signal for a
            clinician to catch.
          </p>
        </motion.div>

        <div className="demo-wrap" style={{ marginTop: 46 }}>
          <motion.div
            className="demo-panel demo-panel--naive"
            initial={{ opacity: 0, x: -24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={viewport}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="demo-head">
              <span className="demo-title">Naive RAG</span>
              <span className="badge badge-danger">
                <span className="dot" />
                clinically wrong
              </span>
            </div>

            <div className="rec rec--stale">
              <span className="rec-label">
                <span className="rec-strike">Potassium</span>
              </span>
              <span className="rec-value rec-strike">2.1 mEq/L</span>
              <span className="rec-date">12 Feb</span>
            </div>
            <div className="rec rec--stale">
              <span className="rec-label">
                <span className="rec-strike">Potassium</span>
              </span>
              <span className="rec-value rec-strike">4.8 mEq/L</span>
              <span className="rec-date">14 Feb</span>
            </div>

            <div className="quote quote--bad">
              &ldquo;The patient&rsquo;s potassium is <strong>2.1 mEq/L</strong>.&rdquo;
              <div className="quote-cite">
                cited: event e4471 — citation is real, value is not
              </div>
            </div>

            <div className="demo-note demo-note--bad">
              Retrieved on semantic similarity alone. No validity state exists in
              the data model, so the stale value wins the tie and is presented as
              fact.
            </div>
          </motion.div>

          <motion.div
            className="demo-panel demo-panel--ours"
            initial={{ opacity: 0, x: 24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={viewport}
            transition={{ duration: 0.6, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="demo-head">
              <span className="demo-title">SoloPrac</span>
              <span className="badge badge-current">
                <span className="dot" />
                verified
              </span>
            </div>

            <div className="rec rec--stale" style={{ opacity: 0.55 }}>
              <span className="rec-label">
                <span className="rec-strike">Potassium</span>
              </span>
              <span className="rec-value rec-strike">2.1 mEq/L</span>
              <span className="badge badge-superseded" style={{ fontSize: 9 }}>
                superseded
              </span>
            </div>
            <div className="rec rec--ok">
              <span className="rec-label">
                <strong>Potassium</strong>
              </span>
              <span className="rec-value" style={{ color: "var(--current)" }}>
                4.8 mEq/L
              </span>
              <span className="badge badge-current" style={{ fontSize: 9 }}>
                current
              </span>
            </div>

            <div className="quote quote--good">
              &ldquo;Potassium is <strong>4.8 mEq/L</strong>, amended from an
              initial 2.1 (specimen re-run).&rdquo;
              <div className="quote-cite">
                cited: event e4489 ✓ current · e4471 ⛔ superseded
              </div>
            </div>

            <div className="demo-note demo-note--good">
              Validity is derived from the supersession graph, never stored. The
              provenance linker rejects any present-tense claim resting on a
              superseded record — regardless of how well it is cited.
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

/* ── three states ───────────────────────────────────────────── */

function Method() {
  const states = [
    {
      cls: "state-card--current",
      symbol: "✓",
      name: "CURRENT",
      body: "No active successor in the graph. Safe to assert as a present-tense fact. This is what a physician means by “the potassium”.",
    },
    {
      cls: "state-card--superseded",
      symbol: "⊘",
      name: "SUPERSEDED",
      body: "A correction points at it. Never asserted as current — but never deleted either, so the change remains auditable.",
    },
    {
      cls: "state-card--asof",
      symbol: "◷",
      name: "AS-OF-T",
      body: "Valid at an instant the clinician explicitly asked about. The same record is current then and superseded now.",
    },
  ];

  return (
    <section className="section" id="method" style={{ background: "var(--bg-2)" }}>
      <div className="container">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={viewport}
          variants={rise}
          custom={0}
        >
          <div className="eyebrow">The method</div>
          <h2 className="section-title">Three states, not two</h2>
          <p className="section-sub">
            Recency decay is a heuristic standing in for a non-temporal
            phenomenon. A lab corrected two hours ago and one corrected two years
            ago are <em>equally invalid</em> for a present-tense question — no
            continuous function of Δt can represent that, and any decay tuned
            hard enough to suppress stale results begins destroying legitimate
            history. Validity comes from graph reachability instead.
          </p>
        </motion.div>

        <div className="states">
          {states.map((s, i) => (
            <motion.div
              key={s.name}
              className={`state-card ${s.cls}`}
              initial={{ opacity: 0, y: 22 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={viewport}
              transition={{ delay: 0.1 * i, duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="state-symbol">{s.symbol}</div>
              <h4>{s.name}</h4>
              <p>{s.body}</p>
            </motion.div>
          ))}
        </div>

        <div className="feature-grid">
          {[
            {
              icon: "⟐",
              title: "Derived, never stored",
              body: "Validity is computed from the supersession DAG on read. A stored status column drifts out of sync with reality; the graph cannot.",
              code: "is_superseded = EXISTS(correction → e)",
            },
            {
              icon: "⊜",
              title: "Provenance-locked generation",
              body: "The model proposes; a linker decides. Every numeric value must appear verbatim in a cited record, or the sentence is dropped rather than hedged.",
              code: "if value not in citation: drop claim",
            },
            {
              icon: "⊙",
              title: "Two-axis verification",
              body: "Axis A checks the claim is entailed by its citation. Axis B checks that citation is valid for the claim's tense. Axis B is the contribution.",
              code: "faithful ∧ temporally_valid",
            },
            {
              icon: "⨯",
              title: "Refuses when unsure",
              body: "Below a confidence threshold the capture pipeline declines to write into the record. Abstention is a feature, not a failure mode.",
              code: "confidence < τ → human review",
            },
          ].map((f, i) => (
            <motion.div
              key={f.title}
              className="feature"
              initial={{ opacity: 0, y: 22 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={viewport}
              transition={{ delay: 0.07 * i, duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
              <div className="feature-code">{f.code}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ── evidence ───────────────────────────────────────────────── */

function Evidence() {
  const rows = [
    ["Superseded value in present tense", "returned as fact", "rejected by the linker"],
    ["Historical query (“as of last March”)", "destroyed by recency decay", "returns the historical value"],
    ["Value invented by the model", "possible", "dropped — not hedged"],
    ["Citation from another patient", "possible", "rejected by the scope assertion"],
    ["Illegible scan", "committed with a guess", "refused, routed to review"],
  ];

  return (
    <section className="section" id="evidence">
      <div className="container">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={viewport}
          variants={rise}
          custom={0}
        >
          <div className="eyebrow">Live behaviour</div>
          <h2 className="section-title">What the system actually does</h2>
          <p className="section-sub">
            Every row below is exercised by the test suite against a live
            database, not asserted in prose.
          </p>
        </motion.div>

        <div className="card" style={{ marginTop: 40, overflow: "hidden" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Scenario", "Naive RAG", "SoloPrac"].map((h, i) => (
                  <th
                    key={h}
                    style={{
                      textAlign: "left",
                      padding: "15px 20px",
                      fontSize: 11.5,
                      textTransform: "uppercase",
                      letterSpacing: "0.06em",
                      color: i === 2 ? "var(--current)" : "var(--text-3)",
                      fontWeight: 700,
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <motion.tr
                  key={r[0]}
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={viewport}
                  transition={{ delay: 0.05 * i, duration: 0.4 }}
                  style={{
                    borderBottom:
                      i === rows.length - 1 ? "none" : "1px solid var(--border)",
                  }}
                >
                  <td style={{ padding: "15px 20px", fontSize: 13.5 }}>{r[0]}</td>
                  <td
                    style={{
                      padding: "15px 20px",
                      fontSize: 13.5,
                      color: "var(--danger)",
                    }}
                  >
                    {r[1]}
                  </td>
                  <td
                    style={{
                      padding: "15px 20px",
                      fontSize: 13.5,
                      color: "var(--current)",
                    }}
                  >
                    {r[2]}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="feature-grid">
          {[
            ["159", "tests passing", "Across validity, provenance, de-identification, retrieval, and the capture gate."],
            ["99%", "coverage of the research core", "validity.py is the module the paper's claim rests on."],
            ["0", "identifiers leave the machine", "Every outbound payload is asserted clean; a failure blocks the request."],
          ].map(([n, label, body], i) => (
            <motion.div
              key={label}
              className="feature"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={viewport}
              transition={{ delay: 0.1 * i, duration: 0.5 }}
            >
              <div
                style={{
                  fontSize: 42,
                  fontWeight: 800,
                  letterSpacing: "-0.04em",
                  color: "var(--current)",
                  lineHeight: 1,
                }}
              >
                {n}
              </div>
              <h3 style={{ marginTop: 10 }}>{label}</h3>
              <p>{body}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ── page ───────────────────────────────────────────────────── */

export default function Landing() {
  return (
    <div className="shell">
      <div className="aurora" />
      <div className="grid-overlay" />
      <div className="content">
        <Nav />
        <Hero />
        <Problem />
        <Method />
        <Evidence />
        <footer className="footer">
          <div className="container footer-inner">
            <div className="brand">
              <div className="brand-mark">S</div>
              <span>SoloPrac</span>
            </div>
            <div>
              A final-year research project · validity-aware clinical retrieval ·
              DPDP-aligned · nothing identifiable leaves the machine
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
