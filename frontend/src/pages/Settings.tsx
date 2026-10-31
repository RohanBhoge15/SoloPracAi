import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import AppShell from "@/components/AppShell";
import PhotoCropper from "@/components/PhotoCropper";
import { api, ApiError, type Doctor } from "@/lib/api";
import { setAccount, useSession } from "@/lib/session";
import "@/pages/settings.css";

const DOCTOR_NAV = [
  { href: "/dashboard", label: "Overview" },
  { href: "/app", label: "Records" },
  { href: "/settings", label: "Settings" },
];
const PATIENT_NAV = [
  { href: "/portal", label: "My documents" },
  { href: "/settings", label: "Settings" },
];

const MAX_PHOTO_MB = 4;

export default function Settings() {
  const session = useSession();
  const isDoctor = session?.account.role === "doctor";

  const [name, setName] = useState(session?.account.display_name ?? "");
  const [phone, setPhone] = useState(session?.account.phone ?? "");
  const [specialty, setSpecialty] = useState("");
  const [clinic, setClinic] = useState("");
  const [nmc, setNmc] = useState("");
  const [state_, setState] = useState("");

  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);
  const [photoBusy, setPhotoBusy] = useState(false);
  const [photoError, setPhotoError] = useState<string | null>(null);
  // The object-storage URL, fetched with the Authorization header. Null
  // means "show the initials", which is both the no-photo state and the
  // state a failed load falls back to.
  const [photoSrc, setPhotoSrc] = useState<string | null>(null);
  // Set when a file is chosen, cleared when the crop is confirmed or
  // cancelled. Nothing is uploaded until it is confirmed.
  const [pendingPhoto, setPendingPhoto] = useState<File | null>(null);

  // Load the picture whenever the account says there is one, and whenever
  // the version changes so a re-uploaded photo is not served from cache.
  const photoPath = session?.account.photo_path;
  const photoVersion = session?.account.photo_version;
  useEffect(() => {
    if (!photoPath) {
      setPhotoSrc(null);
      return;
    }
    let live = true;
    api
      .photoUrl()
      .then(({ url }) => {
        if (live) setPhotoSrc(url);
      })
      .catch(() => {
        // No picture is a normal state, not an error worth shouting about.
        if (live) setPhotoSrc(null);
      });
    return () => {
      live = false;
    };
  }, [photoPath, photoVersion]);

  useEffect(() => {
    if (!isDoctor) return;
    // The doctor profile is a separate resource from the account, so it
    // is fetched rather than assumed to be empty.
    api
      .me()
      .then(({ doctor }) => {
        if (!doctor) return;
        setSpecialty(doctor.specialty ?? "");
        setClinic(doctor.clinic_name ?? "");
        setNmc(doctor.nmc_number ?? "");
        setState(doctor.registration_state ?? "");
      })
      .catch(() => {
        /* the account fields are still editable without it */
      });
  }, [isDoctor]);

  async function saveProfile(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      const account = await api.updateProfile({
        display_name: name.trim(),
        phone: phone.trim() || null,
      });
      setAccount(account);

      if (isDoctor) {
        const payload: Partial<Doctor> = {
          specialty: specialty.trim() || null,
          clinic_name: clinic.trim() || null,
          nmc_number: nmc.trim() || null,
          registration_state: state_.trim() || null,
        };
        await api.updateDoctor(payload);
      }
      setSaved(true);
      window.setTimeout(() => setSaved(false), 2600);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save.");
    } finally {
      setBusy(false);
    }
  }

  async function onPhoto(file: File) {
    setPhotoError(null);

    // Checked here as well as server-side so an oversized phone photo
    // fails instantly instead of after a slow upload.
    if (file.size > MAX_PHOTO_MB * 1024 * 1024) {
      setPhotoError(`That image is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is ${MAX_PHOTO_MB} MB.`);
      return;
    }
    if (!file.type.startsWith("image/")) {
      setPhotoError("That file is not an image.");
      return;
    }

    // Nothing is uploaded yet. Selecting a file opens the crop dialog, and
    // only a confirmed crop is sent, so a cancelled crop leaves no object
    // in storage and no row in the database.
    setPendingPhoto(file);
  }

  /** Called with the cropped blob once the dialog is confirmed. */
  async function uploadCropped(blob: Blob) {
    setPhotoBusy(true);
    setPhotoError(null);
    try {
      const cropped = new File([blob], "avatar.jpg", { type: blob.type });
      const account = await api.uploadPhoto(cropped);
      setAccount(account);
      // Pull the new URL straight away. Waiting for the effect to re-run
      // left a window where the avatar showed the *previous* photo, or the
      // initials, after a successful upload - which reads as "nothing
      // happened" all over again.
      const { url } = await api.photoUrl();
      setPhotoSrc(url);
      setPhotoError(null);
    } catch (err) {
      setPhotoError(
        err instanceof ApiError ? err.message : "Could not upload that.",
      );
    } finally {
      setPhotoBusy(false);
      if (fileRef.current) fileRef.current.value = "";
      setPendingPhoto(null);
    }
  }

  const account = session?.account;
  const initials = (account?.display_name ?? "?")
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");

  return (
    <AppShell items={isDoctor ? DOCTOR_NAV : PATIENT_NAV}>
      <h1 className="dash-title">Settings</h1>
      <p className="dash-sub">
        {isDoctor
          ? "Your name and registration details are printed on every document you issue."
          : "Your details, so your doctors can reach you."}
      </p>

      <div className="set-grid">
        <section className="card set-card">
          <h2 className="set-h2">Profile picture</h2>
          <div className="set-photo">
            {photoSrc ? (
              <img
                src={photoSrc}
                alt=""
                className="set-avatar"
                onError={() => {
                  // Fall back to the initials rather than hiding the image.
                  // Hiding it left an empty hole that read as "the upload
                  // did nothing", which is exactly what happened when this
                  // pointed at an authenticated endpoint: the <img> 401'd,
                  // the fallback was already unmounted, and the picture the
                  // user had just uploaded became invisible.
                  setPhotoSrc(null);
                }}
              />
            ) : (
              <span className="set-avatar set-avatar-fallback">{initials}</span>
            )}

            <div className="set-photo-actions">
              <input
                ref={fileRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void onPhoto(file);
                }}
                className="visually-hidden"
                id="photo-input"
              />
              <label htmlFor="photo-input" className="btn btn-ghost btn-sm">
                {photoBusy ? <span className="spinner" /> : "Choose image"}
              </label>
              <p className="set-hint">
                JPEG, PNG or WebP, up to {MAX_PHOTO_MB} MB.
              </p>
              {photoError && <p className="set-err">{photoError}</p>}
            </div>
          </div>
        </section>

        <section className="card set-card">
          <h2 className="set-h2">Your details</h2>
          <form onSubmit={saveProfile} className="set-form">
            <label className="field">
              <span>Full name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </label>

            <label className="field">
              <span>Email</span>
              <input value={account?.email ?? ""} disabled />
              <small>
                {isDoctor
                  ? "Your identity. Changing it needs a verification step that is not built yet."
                  : "This is what links you to records your doctors already hold."}
              </small>
            </label>

            <label className="field">
              <span>Phone</span>
              <input
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                placeholder="+91 ..."
              />
            </label>

            {isDoctor && (
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
                    <span>Registration state</span>
                    <input
                      value={state_}
                      onChange={(e) => setState(e.target.value)}
                      placeholder="Kerala"
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

                <label className="field">
                  <span>NMC / medical registration number</span>
                  <input
                    value={nmc}
                    onChange={(e) => setNmc(e.target.value)}
                    placeholder="A-12345"
                  />
                  <small>
                    Printed in the signature block of every prescription.
                  </small>
                </label>
              </>
            )}

            {error && (
              <p className="auth-error" role="alert">
                {error}
              </p>
            )}
            {saved && (
              <motion.p
                className="auth-notice"
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
              >
                Saved.
              </motion.p>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              disabled={busy}
            >
              {busy ? <span className="spinner" /> : "Save changes"}
            </button>
          </form>
        </section>
      </div>

      {pendingPhoto && (
        <PhotoCropper
          file={pendingPhoto}
          onCancel={() => {
            setPendingPhoto(null);
            if (fileRef.current) fileRef.current.value = "";
          }}
          onConfirm={(result) => {
            setPhotoSrc(result.previewUrl);
            void uploadCropped(result.blob);
          }}
        />
      )}
    </AppShell>
  );
}
