/**
 * REAL-DATA research check for SoloPrac.
 *
 * Goal: behave like a *realistic* patient record (multiple labs, a
 * correction, a stopped medication, an imaging note) and verify the
 * research semantics persist end-to-end:
 *
 *   1. An event superseded by a later correction shows up with
 *      validity="superseded" in the timeline.
 *   2. An as-of query against the ORIGINAL instant returns the original
 *      (pre-correction) value.
 *   3. A present-tense chat question answers with the CURRENT record's
 *      value, not the superseded one.
 *   4. A historical chat question (as_of = before correction) answers
 *      with the originally recorded value.
 *   5. Citations in chat responses carry validity labels.
 *
 * Run:  node solum/_e2e/research_verify.mjs
 */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { existsSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
mkdirSync(join(HERE, "shots"), { recursive: true });

const BASE = process.env.BASE || "http://localhost:8080";
const tag = randomUUID().slice(0, 10);

function findChromium() {
  const base = join(process.env.LOCALAPPDATA || "", "ms-playwright");
  for (const rev of ["chromium-1234", "chromium-1228", "chromium-1208", "chromium-1217"]) {
    const candidates = [
      join(base, `${rev}/chrome-win64/chrome.exe`),
      join(base, `${rev}/chrome-win/chrome.exe`),
      join(base, `${rev}/chromium-win/chrome.exe`),
    ];
    for (const c of candidates) {
      try { if (existsSync(c)) return c; } catch { /* ignore */ }
    }
  }
  return undefined;
}

let browser;

const results = [];
function log(ok, label, detail = "") {
  results.push({ ok, label, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
}

const ok = async (page, url, options) => {
  const res = await page.evaluate(async ({ url, options }) => {
    const token = localStorage.getItem("solum.token");
    const headers = {
      Authorization: `Bearer ${token}`,
      ...(options?.headers || {}),
    };
    return fetch(url, { ...options, headers }).then(async (r) => ({
      status: r.status,
      body: await r.json().catch(() => ({})),
    }));
  }, { url, options });
  return res;
};

async function main() {
  browser = await chromium.launch({ executablePath: findChromium() });
  const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });

  // 1) Register doctor + add patient via UI to have a token in localStorage
  await page.goto(`${BASE}/register`, { waitUntil: "networkidle" });
  await page.getByRole("tab", { name: "Doctor Registration" }).click();
  await page.locator('input[placeholder="As it appears on your prescriptions"]').first().fill(`Dr Research Verify ${tag}`);
  await page.locator('input[placeholder="you@example.com"]').first().fill(`dr.${tag}@solum.test`);
  await page.locator('input[placeholder="At least 8 characters"]').first().fill("correct-horse-battery");
  await page.locator('input[placeholder="General Medicine"]').first().fill("General Medicine");
  await page.locator('button[type="submit"]').first().click();
  try { await page.waitForURL("**/dashboard", { timeout: 30000 }); } catch (e) {
    log(false, "doctor registered and lands on /dashboard", String(e)); await browser.close(); process.exit(1);
  }

  // Create a patient via API so profiles exist for timeline queries
  const origEvent = await createDoctorPatientAndBaseEvents(page);
  if (!origEvent.ok) { log(false, "seed patient + events", origEvent.error); await browser.close(); process.exit(1); }
  const { patientId, origHba1cId } = origEvent;
  console.log(`  patient id: ${patientId.slice(0, 8)}`);

  // â”€â”€ Supersession: correct the HbA1c â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const correction = await page.evaluate(async ({ patientId, origHba1cId }) => {
    const token = localStorage.getItem("solum.token");
    const body = {
      event_type: "lab",
      valid_time: new Date(Date.now() - 60 * 86400000).toISOString(),
      payload: { test: "HbA1c", value: 7.0, unit: "%", ref_low: 4.0, ref_high: 5.7, flag: "high", note: "Corrected after repeat lab" },
      source_type: "correction",
      supersedes_id: origHba1cId,
    };
    return fetch(`/api/v1/patients/${patientId}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
    }).then(async (r) => ({ status: r.status, body: await r.json().catch(() => ({})) }));
  }, { patientId, origHba1cId });
  log(correction.status === 201, "correction event created via API", `status ${correction.status}`);

  // â”€â”€ A. Timeline semantics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const timeline = (await ok(page, `/api/v1/patients/${patientId}/timeline`)).body;
  log(timeline.current.some((e) => e.payload?.test === "HbA1c" && e.validity !== "superseded"),
      "timeline has a current HbA1c record", "");

  const pairs = timeline.pairs || [];
  const hba1cPair = pairs.find((p) => p.superseded?.payload?.test === "HbA1c");
  log(!!hba1cPair, "superseded HbA1c appears in a superseded pair", hba1cPair ? `superseded id=${hba1cPair.superseded.id.slice(0, 8)}` : "no pair");
  log(hba1cPair && hba1cPair.correction?.payload?.value === 7.0, "the correction is the paired successor", hba1cPair ? `value=${hba1cPair.correction?.payload?.value}` : "");

  // â”€â”€ B. as-of view returns the original record â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const pastIso = new Date(Date.now() - 180 * 86400000).toISOString();
  const asof = (await ok(page, `/api/v1/patients/${patientId}/as-of/${encodeURIComponent(pastIso)}`)).body;
  const asofHba1c = (asof.events || []).find((e) => e.payload?.test === "HbA1c");
  log(!!(asofHba1c && asofHba1c.payload?.value === 8.9),
      "as-of view (180d ago) exposes the original event", asofHba1c ? `value=${asofHba1c.payload?.value}` : "missing");

  const nowIso = new Date().toISOString();
  const nowAsof = (await ok(page, `/api/v1/patients/${patientId}/as-of/${encodeURIComponent(nowIso)}`)).body;
  const nowHba1c = (nowAsof.events || []).find((e) => e.payload?.test === "HbA1c");
  log(!!(nowHba1c && nowHba1c.payload?.value === 7.0 && nowHba1c.validity === "current"),
      "as-of view (now) shows the correction as current", nowHba1c ? `value=${nowHba1c.payload?.value}, validity=${nowHba1c.validity}` : "missing");
  log(!(nowAsof.events || []).some((e) => e.payload?.test === "HbA1c" && e.payload?.value === 8.9),
      "as-of view (now) no longer exposes the superseded HbA1c", "");

  // â”€â”€ C. Present-tense chat answers with the CURRENT record â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const latest = (await ok(page, `/api/v1/patients/${patientId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_id: patientId, query: "What is the latest HbA1c?" }),
  })).body;
  log(latest.answer && /7(?:[.,]0)?\s*%/i.test(latest.answer), "chat names the corrected value", latest.answer?.slice(0, 60).replace(/\n/g, " "));
  log(!(latest.answer && /8\.9/.test(latest.answer)), "chat does NOT answer with the superseded value", "");
  log((latest.citations || []).some((c) => c.validity === "current"), "citations are labelled current", "");

  // â”€â”€ D. Historical chat answers with the original record â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const histIso = new Date(Date.now() - 180 * 86400000).toISOString();
  const hist = (await ok(page, `/api/v1/patients/${patientId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      query: "What was the HbA1c around 180 days ago?",
      as_of: histIso,
    }),
  })).body;
  log(hist.answer && /8\.9/.test(hist.answer), "history chat names the original value", hist.answer?.slice(0, 60).replace(/\n/g, " "));
  log(!(hist.answer && /7(?:[.,]0)?\s*%/i.test(hist.answer)), "history chat does NOT leak the later correction", "");

  await page.screenshot({ path: join(HERE, "shots", "research-verify.png"), fullPage: true });
  await browser.close();

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  if (failed.length) process.exit(1);
  process.exit(0);
}

async function createDoctorPatientAndBaseEvents(page) {
  // Create patient via API to spend less time in the UI for this check
  const created = await page.evaluate(async ({ tag }) => {
    const token = localStorage.getItem("solum.token");
    const p = await fetch(`/api/v1/patients`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({
        display_name: `Research Verify ${tag}`,
        age: 52,
        sex: "male",
        email: `patient-${tag}@example.com`,
        phone: "9000000000",
      }),
    }).then(async (r) => ({ status: r.status, body: await r.json().catch(() => ({})) }));
    if (p.status !== 201) return { ok: false, error: `create patient status ${p.status}` };
    const patientId = p.body.id;

    const now = Date.now();
    const d = (agoDays) => new Date(now - agoDays * 86400000).toISOString();
    const events = [
      {
        event_type: "lab",
        valid_time: d(200),
        payload: { test: "HbA1c", value: 8.9, unit: "%", ref_low: 4.0, ref_high: 5.7, flag: "high" },
        source_type: "manual",
      },
      {
        event_type: "lab",
        valid_time: d(190),
        payload: { test: "Potassium", value: 4.8, unit: "mEq/L" },
        source_type: "manual",
      },
      {
        event_type: "medication",
        valid_time: d(180),
        payload: { drug: "Metformin", dose: "500 mg", frequency: "twice daily", status: "active" },
        source_type: "manual",
      },
      {
        event_type: "medication",
        valid_time: d(150),
        payload: { drug: "Lisinopril", dose: "5 mg", frequency: "once daily", status: "stopped", reason: "Cough" },
        source_type: "manual",
      },
      {
        event_type: "vital",
        valid_time: d(7),
        payload: { measurement: "Blood pressure", value: "128/82", unit: "mmHg" },
        source_type: "manual",
      },
      {
        event_type: "diagnosis",
        valid_time: d(30),
        payload: { diagnosis: "Type 2 Diabetes Mellitus", icd10: "E11.9" },
        source_type: "manual",
      },
      {
        event_type: "note",
        valid_time: d(2),
        payload: { note: "Patient reports mild dizziness on standing", author: "Dr Research Verify" },
        source_type: "manual",
      },
    ];
    const bulk = await fetch(`/api/v1/patients/${patientId}/events/bulk`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(events),
    }).then(async (r) => ({ status: r.status, body: await r.json().catch(() => ({})) }));
    if (bulk.status !== 201) return { ok: false, error: `bulk events status ${bulk.status}` };
    const origHba1c = bulk.body.find((e) => e.payload?.test === "HbA1c" && e.payload?.value === 8.9);
    return { ok: true, patientId, origHba1cId: origHba1c.id };
  }, { tag });
  return created;
}

main().catch((err) => { console.error(err); process.exit(1); });
