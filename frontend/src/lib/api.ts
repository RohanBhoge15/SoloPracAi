/* ============================================================
   Typed API client.

   Same-origin in dev via the Vite proxy, and same-origin in
   production behind nginx, so there is no CORS and no base URL
   to configure.
   ============================================================ */

export type Validity = "current" | "superseded";

export interface EventOut {
  id: string;
  event_type: string;
  source_type: string;
  valid_time: string;
  recorded_at: string;
  payload: Record<string, unknown>;
  confidence: number | null;
  supersedes_id: string | null;
  content_hash: string;
  is_superseded: boolean;
  validity: Validity;
}

export interface SupersededPair {
  superseded: EventOut;
  correction: EventOut;
}

export interface Timeline {
  patient_id: string;
  current: EventOut[];
  history: EventOut[];
  pairs: SupersededPair[];
  stats: { total: number; current: number; superseded: number; corrections: number };
}

export interface Patient {
  id: string;
  display_name: string;
  age: number | null;
  phone: string | null;
  email: string | null;
  created_at: string;
  /** Present once the patient has uploaded a picture. The backend
   *  mirrors it onto the patient row, so a doctor sees it without the
   *  patient doing anything. Null while they have not. */
  photo_path?: string | null;
}

export interface Citation {
  event_id: string;
  text: string;
  valid_time: string;
  validity: Validity;
  event_type: string;
}

export interface Suppressed {
  text: string;
  reason: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  suppressed: Suppressed[];
  dropped: Suppressed[];
  intent: string;
  intent_confidence: number;
  degraded: boolean;
  latency_ms: number;
  model: string;
  has_stale_assertion: boolean;
}

export interface Health {
  status: string;
  service: string;
  version: string;
  llm_configured: boolean;
  llm_model: string;
}

export interface EventCreate {
  event_type: string;
  valid_time: string;
  payload: Record<string, unknown>;
  source_type?: string;
  supersedes_id?: string | null;
}

export interface CaptureResult {
  document: {
    id: string;
    filename: string;
    ocr_confidence: number | null;
    committed: boolean;
    review_reason: string | null;
    created_at: string;
  };
  extracted: { rows?: Array<Record<string, unknown>>; sample_date?: string | null };
  confidence: number;
  committed: boolean;
  reason: string | null;
  coverage_note: string;
}

/* ── auth ───────────────────────────────────────────────────── */

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

export interface Doctor {
  id: string;
  account_id: string;
  specialty: string | null;
  clinic_name: string | null;
  nmc_number: string | null;
  registration_state: string | null;
  consult_fee: number | null;
  signature_path: string | null;
}

export interface SessionPayload {
  token: string;
  account: Account;
  doctor: Doctor | null;
  /** Records this account inherited by email match, on registration. */
  linked_records: number;
  patients_visible: number;
}

/* ── dashboard ──────────────────────────────────────────────── */

export interface Kpis {
  patients: number;
  patients_added_7d: number;
  records: number;
  records_added_today: number;
  records_added_7d: number;
  documents: number;
  prescriptions: number;
  reports: number;
  certificates: number;
  invoices: number;
  documents_sent_today: number;
  awaiting_review: number;
}

export interface PatientRow {
  id: string;
  display_name: string;
  age: number | null;
  phone: string | null;
  email: string | null;
  photo_path: string | null;
  record_count: number;
  last_activity: string | null;
  created_at: string;
}

export interface Dashboard {
  doctor_name: string;
  specialty: string | null;
  clinic_name: string | null;
  kpis: Kpis;
  recent_patients: PatientRow[];
  total_patients: number;
}

/* ── patient portal ─────────────────────────────────────────── */

export interface InboxItem {
  document_id: string;
  kind: string;
  title: string;
  reference: string;
  size_bytes: number | null;
  issued_at: string | null;
  read: boolean;
  superseded: boolean;
  doctor_name: string | null;
  doctor_specialty: string | null;
}

export interface PatientRecordRow {
  patient_id: string;
  display_name: string;
  doctor_name: string | null;
  specialty: string | null;
  /** The doctor's account, so their picture can be requested. Absent for
   *  a record with no doctor attached. */
  doctor_account_id?: string | null;
  /** A hint, so a portal full of doctors without pictures does not fire
   *  a request per row. */
  doctor_has_photo?: boolean;
  event_count: number;
  since: string | null;
}

export interface OpenedDocument {
  url: string;
  filename: string;
  content_type: string;
  size_bytes: number | null;
  expires_in: number;
}

/* ── notifications ──────────────────────────────────────────── */

export interface Notification {
  id: string;
  kind: string;
  title: string;
  body: string;
  payload: Record<string, unknown>;
  created_at: string;
  read: boolean;
}

/* ── documents ──────────────────────────────────────────────── */

export interface DocumentOut {
  id: string;
  patient_id: string;
  doctor_id: string | null;
  kind: string;
  title: string | null;
  description: string | null;
  object_key: string | null;
  content_type: string | null;
  size_bytes: number | null;
  content: Record<string, unknown>;
  issued_to_patient: boolean;
  issued_at: string | null;
  read_at: string | null;
  version: number;
  created_at: string;
  committed: boolean;
  review_reason: string | null;
  superseded_by_id: string | null;
}

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/* ── auth plumbing ──────────────────────────────────────────── */

import { clearSession, getToken } from "./session";

/**
 * Called when the API rejects our token.
 *
 * Clearing here rather than in each caller means an expired session
 * cannot leave one screen showing stale clinical data while another
 * has already signed out. It is a module-level callback so `session`
 * does not have to import `api`, which would be circular.
 */
let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(fn: (() => void) | null) {
  onUnauthorized = fn;
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseError(res: Response): Promise<string> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    // FastAPI puts a string in `detail` for hand-raised errors and a
    // list of objects in it for request-validation errors.
    if (typeof body?.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
      detail = body.detail[0].msg;
    }
  } catch {
    /* non-JSON error body */
  }
  return detail;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...authHeaders(),
    ...((init?.headers as Record<string, string>) ?? {}),
  };

  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch {
    throw new ApiError(
      "Cannot reach the SoloPrac API. Is it running on port 8000?",
      0,
    );
  }

  if (!res.ok) {
    if (res.status === 401) {
      // The token is gone or expired. Sign out once, globally.
      clearSession();
      onUnauthorized?.();
    }
    throw new ApiError(await parseError(res), res.status);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** Multipart upload. The browser sets the boundary, so no Content-Type. */
async function upload<T>(path: string, form: FormData): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", body: form, headers });
  } catch {
    throw new ApiError("Upload failed - is the API running?", 0);
  }
  if (!res.ok) {
    if (res.status === 401) {
      clearSession();
      onUnauthorized?.();
    }
    throw new ApiError(await parseError(res), res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  config: () =>
    request<{
      app_name: string;
      llm_enabled: boolean;
      llm_model: string;
      retrieval_k: number;
      ocr_confidence_threshold: number;
    }>("/api/v1/config"),

  /* ── auth ─────────────────────────────────────────────── */

  login: (email: string, password: string) =>
    request<SessionPayload>("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  registerDoctor: (body: {
    email: string;
    password: string;
    display_name: string;
    phone?: string | null;
    specialty?: string | null;
    clinic_name?: string | null;
    nmc_number?: string | null;
    registration_state?: string | null;
  }) =>
    request<SessionPayload>("/api/v1/auth/register/doctor", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  registerPatient: (body: {
    email: string;
    password: string;
    display_name: string;
    phone?: string | null;
  }) =>
    request<SessionPayload>("/api/v1/auth/register/patient", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  me: () => request<{ account: Account; doctor: Doctor | null }>("/api/v1/auth/me"),

  updateProfile: (body: {
    display_name?: string;
    phone?: string | null;
  }) =>
    request<Account>("/api/v1/auth/me/profile", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  updateDoctor: (body: Partial<Doctor> & { display_name?: string }) =>
    request<Doctor>("/api/v1/auth/me/doctor", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  uploadPhoto: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return upload<Account>("/api/v1/auth/me/photo", form);
  },

  /**
   * A presigned URL for the signed-in account's profile picture.
   *
   * The `<img>` tag cannot send an Authorization header, so it cannot be
   * pointed straight at an authenticated endpoint - that 401s and the
   * avatar renders as blank space. Fetch the URL here, where the header
   * can be attached, and give the `<img>` the object-storage URL, which
   * needs no auth.
   */
  photoUrl: () =>
    request<{ url: string; version: number }>("/api/v1/documents/photo/url"),

  /**
   * A presigned URL for another account's avatar, e.g. a patient's own
   * doctor. Null when they have not set a picture, which is normal and
   * not an error.
   */
  avatarUrl: async (accountId: string): Promise<string | null> => {
    try {
      const { url } = await request<{ url: string }>(
        `/api/v1/documents/avatars/${accountId}`,
      );
      return url;
    } catch {
      return null;
    }
  },

  /**
   * A presigned URL for a patient's own picture, for their doctor.
   *
   * Null when the patient has not uploaded one. Never throws for that
   * case - a missing picture is normal, not an error.
   */
  patientPhotoUrl: async (patientId: string): Promise<string | null> => {
    try {
      const { url } = await request<{ url: string }>(
        `/api/v1/patients/${patientId}/photo`,
      );
      return url;
    } catch {
      return null;
    }
  },

  /* ── dashboard ───────────────────────────────────────── */

  dashboard: (search = "", limit = 8) => {
    const q = new URLSearchParams({ limit: String(limit) });
    if (search.trim()) q.set("search", search.trim());
    return request<Dashboard>(`/api/v1/dashboard?${q.toString()}`);
  },

  /* ── patients ────────────────────────────────────────── */

  listPatients: () => request<Patient[]>("/api/v1/patients"),

  createPatient: (body: {
    display_name: string;
    age?: number | null;
    phone?: string | null;
    email?: string | null;
    address?: string | null;
  }) =>
    request<Patient>("/api/v1/patients", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  deletePatient: (patientId: string) =>
    request<void>(`/api/v1/patients/${patientId}`, { method: "DELETE" }),

  patientStats: (patientId: string) =>
    request<Record<string, unknown>>(`/api/v1/patients/${patientId}/stats`),

  createEvent: (patientId: string, body: EventCreate) =>
    request<EventOut>(`/api/v1/patients/${patientId}/events`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  timeline: (patientId: string) =>
    request<Timeline>(`/api/v1/patients/${patientId}/timeline`),

  chat: (patientId: string, query: string, asOf?: string) =>
    request<ChatResponse>(`/api/v1/patients/${patientId}/chat`, {
      method: "POST",
      body: JSON.stringify({
        patient_id: patientId,
        query,
        as_of: asOf ?? null,
      }),
    }),

  asOf: (patientId: string, instant: string) =>
    request<{ patient_id: string; instant: string; events: EventOut[] }>(
      `/api/v1/patients/${patientId}/as-of/${encodeURIComponent(instant)}`,
    ),

  capture: (
    patientId: string,
    file: File,
    opts: {
      eventType?: "lab" | "imaging";
      /** Study description. Imaging only; a lab sheet is read, not described. */
      caption?: string;
      /** A radiologist's or doctor's own description of the study. */
      findings?: string;
      /**
       * When the study was *taken*, ISO-8601. Defaults to now, which files an
       * old radiograph as current -- so the caller should set it when known.
       */
      validTime?: string;
    } = {},
  ) => {
    const form = new FormData();
    form.append("file", file);
    form.append("event_type", opts.eventType ?? "lab");
    if (opts.caption) form.append("caption", opts.caption);
    if (opts.findings) form.append("findings", opts.findings);
    if (opts.validTime) form.append("valid_time", opts.validTime);
    return upload<CaptureResult>(
      `/api/v1/patients/${patientId}/capture`,
      form,
    );
  },

  /**
   * Find this patient's stored studies by what they *depict*.
   *
   * The description is embedded into BiomedCLIP's joint space alongside the
   * stored pixels, so the ranking comes from the image rather than from the
   * caption text. Scoped to one patient by `patient_id`, and the caller's
   * own chart only.
   */
  imageSearch: (patientId: string, description: string, limit = 5) =>
    request<{
      query: string;
      hits: Array<{ event_id: string; caption: string; score: number }>;
      degraded: boolean;
    }>(
      `/api/v1/semantic/image-search?patient_id=${patientId}` +
        `&description=${encodeURIComponent(description)}&limit=${limit}`,
      { method: "POST" },
    ),

  /**
   * Fetch a document's bytes for the in-app viewer.
   *
   * `<img src>` and `<iframe src>` cannot send an Authorization header, and
   * these documents carry the patient's name, age and phone on the page. So
   * the bytes are fetched with the token and handed to the viewer as a blob
   * URL, which keeps the authorisation on the request rather than in a
   * presigned link that can be forwarded.
   *
   * `/raw` sends `Content-Disposition: attachment`, which would make a
   * browser download the file instead of displaying it, so it is fetched
   * through JS rather than pointed at directly. The server's content type is
   * used to decide how to render, and an unknown type falls back to a
   * download rather than being displayed as if it were safe to render.
   */
  documentBlob: async (documentId: string) => {
    const res = await fetch(`/api/v1/documents/${documentId}/raw`, {
      headers: authHeaders(),
    });
    if (!res.ok) {
      if (res.status === 401) {
        clearSession();
        onUnauthorized?.();
      }
      throw new ApiError(await parseError(res), res.status);
    }
    // The body can be read once, so the blob is materialised before the
    // object URL is made from it.
    const blob = await res.blob();
    return {
      blob,
      contentType: res.headers.get("content-type") ?? "application/octet-stream",
      url: URL.createObjectURL(blob),
    };
  },

  /* ── patient portal ──────────────────────────────────── */

  inbox: () => request<InboxItem[]>("/api/v1/patient/inbox"),

  myRecords: () => request<PatientRecordRow[]>("/api/v1/patient/records"),

  openDocument: (documentId: string) =>
    request<OpenedDocument>(
      `/api/v1/patient/documents/${documentId}/open`,
    ),

  /* ── notifications ───────────────────────────────────── */

  notifications: (limit = 30) =>
    request<Notification[]>(`/api/v1/notifications?limit=${limit}`),

  unreadCount: () =>
    request<{ unread: number; total: number }>(
      "/api/v1/notifications/unread-count",
    ),

  markRead: (ids: string[]) =>
    request<{ marked: number }>("/api/v1/notifications/read", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),

  /* ── documents ───────────────────────────────────────── */

  documentsForPatient: (patientId: string) =>
    request<DocumentOut[]>(`/api/v1/documents/patient/${patientId}`),

  documentCapabilities: () =>
    request<{ available: boolean; reason?: string; backend?: string }>(
      "/api/v1/documents/capabilities",
    ),

  createPrescription: (body: Record<string, unknown>) =>
    request<DocumentOut>("/api/v1/documents/prescription", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  createCertificate: (body: Record<string, unknown>) =>
    request<DocumentOut>("/api/v1/documents/certificate", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  createInvoice: (body: Record<string, unknown>) =>
    request<DocumentOut>("/api/v1/documents/invoice", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  createWeeklyReport: (body: Record<string, unknown>) =>
    request<DocumentOut>("/api/v1/documents/weekly-report", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  downloadDocument: (documentId: string) =>
    request<{ url: string; filename: string; expires_in: number }>(
      `/api/v1/documents/${documentId}/download`,
    ),

  suggestDrugs: (q: string, limit = 8) => {
    const params = new URLSearchParams({
      q: q.trim(),
      limit: String(limit),
    });
    return request<{ query: string; suggestions: string[]; total_names: number }>(
      `/api/v1/drugs/suggest?${params.toString()}`,
    );
  },
};

export { ApiError };
