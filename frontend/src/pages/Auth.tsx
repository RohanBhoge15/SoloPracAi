import { useState, type FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api, ApiError } from "@/lib/api";
import { setSession, landingFor } from "@/lib/session";
import "@/pages/auth.css";

type Mode = "signin" | "doctor" | "patient";
type Role = "doctor" | "patient";

const COPY: Record<Mode, { title: string; sub: string; cta: string }> = {
  signin: {
    title: "Welcome back",
    sub: "Sign in to your records.",
    cta: "Sign in",
  },
  doctor: {
    title: "Set up your practice",
    sub: "Your email is your identity. Prescriptions you issue are watermarked with your name, and patients see who sent them.",
    cta: "Create doctor account",
  },
  patient: {
    title: "Access your records",
    sub: "Use the same email your doctor has on file and your records appear automatically.",
    cta: "Create patient account",
  },
};

export default function Auth({ initialMode = "signin" }: { initialMode?: Mode }) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [clinic, setClinic] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const isRegister = mode !== "signin";
  const role: Role = mode === "doctor" ? "doctor" : "patient";

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      const result =
        mode === "signin"
          ? await api.login(email, password)
          : role === "doctor"
            ? await api.registerDoctor({
              email,
              password,
              display_name: name,
              phone: phone || null,
              specialty: specialty || null,
              clinic_name: clinic || null,
            })
            : await api.registerPatient({
              email,
              password,
              display_name: name,
              phone: phone || null,
            });

      setSession({ token: result.token, account: result.account });

      // Registration can silently merge a walk-in record into the new
      // account. Saying so is the whole point of requiring email, so it
      // is surfaced rather than left for the user to discover.
      if (isRegister && result.linked_records > 0) {
        setNotice(
          `Found ${result.linked_records} existing record${result.linked_records === 1 ? "" : "s"
          } under this email. They are now linked to your account.`,
        );
        window.setTimeout(() => {
          window.location.href = landingFor({
            token: result.token,
            account: result.account,
          });
        }, 2600);
        return;
      }

      window.location.href = landingFor({
        token: result.token,
        account: result.account,
      });
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong. Please try again.",
      );
      setBusy(false);
    }
  }

  const copy = COPY[mode];

  return (
    <div className="auth">
      <div className="aurora" />
      <div className="grid-overlay" />

      <div className="auth-card card">
        <a href="/" className="auth-brand">
          <span className="brand-mark">S</span>
          <span>SoloPrac</span>
        </a>

        <AnimatePresence mode="wait">
          <motion.div
            key={mode}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            <h1 className="auth-title">{copy.title}</h1>
            <p className="auth-sub">{copy.sub}</p>
          </motion.div>
        </AnimatePresence>

        <div className="auth-tabs" role="tablist">
          {(
            [
              ["signin", "Sign in"],
              ["doctor", "Doctor Registration"],
              ["patient", "Patient Registration"],
            ] as [Mode, string][]
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={mode === value}
              className={`auth-tab ${mode === value ? "on" : ""}`}
              onClick={() => {
                setMode(value);
                setError(null);
                setNotice(null);
              }}
            >
              {label}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="auth-form">
          {isRegister && (
            <label className="field">
              <span>Full name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="As it appears on your prescriptions"
                autoComplete="name"
                required
              />
            </label>
          )}

          <label className="field">
            <span>Email</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
              required
            />
            {isRegister && (
              <small>
                {role === "doctor"
                  ? "Required. Your prescriptions are watermarked with your name."
                  : "Use the email your doctor has on file and your records link automatically."}
              </small>
            )}
          </label>

          <label className="field">
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={isRegister ? "At least 8 characters" : ""}
              autoComplete={isRegister ? "new-password" : "current-password"}
              minLength={isRegister ? 8 : undefined}
              required
            />
          </label>

          {mode === "doctor" && (
            <>
              <div className="field-row">
                <label className="field">
                  <span>Specialty</span>
                  <input
                    value={specialty}
                    onChange={(e) => setSpecialty(e.target.value)}
                    placeholder="General Medicine"
                  />
                </label>
                <label className="field">
                  <span>Phone</span>
                  <input
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    placeholder="+91 ..."
                  />
                </label>
              </div>
              <label className="field">
                <span>Clinic name</span>
                <input
                  value={clinic}
                  onChange={(e) => setClinic(e.target.value)}
                  placeholder="Printed on every document"
                />
              </label>
            </>
          )}

          {mode === "patient" && (
            <label className="field">
              <span>Phone (optional)</span>
              <input
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                placeholder="+91 ..."
              />
            </label>
          )}

          {error && (
            <motion.p
              className="auth-error"
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              role="alert"
            >
              {error}
            </motion.p>
          )}
          {notice && (
            <motion.p
              className="auth-notice"
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
            >
              {notice}
            </motion.p>
          )}

          <button
            type="submit"
            className="btn btn-primary auth-submit"
            disabled={busy}
          >
            {busy ? <span className="spinner" /> : copy.cta}
          </button>
        </form>

        <p className="auth-foot">
          <a href="/">Back to home</a>
        </p>
      </div>
    </div>
  );
}
