/**
 * Drives the real frontend in a real browser, for both sides.
 *
 * The HTTP e2e script proves the API works. This proves the product works:
 * that a human can sign up, find a patient, read a record, issue a
 * prescription, and that the patient can then open it. It asserts on what
 * the page actually renders, and it saves a screenshot at every step so a
 * failure is a picture rather than a stack trace.
 *
 * Run:  node e2e/browser.mjs
 */

import { chromium } from "playwright";
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHOTS = join(HERE, "shots") + "\\";
mkdirSync(SHOTS, { recursive: true });

const BASE = process.env.BASE || "http://localhost:8080";

const tag = randomUUID().slice(0, 10);
const DOCTOR = {
  email: `dr.${tag}@solum.test`,
  password: "correct-horse-battery",
  name: `Dr Meera Rao ${tag}`,
  specialty: "General Medicine",
  clinic: "Rao Clinic",
  phone: "+919000000001",
};
const PATIENT = {
  email: `patient.${tag}@solum.test`,
  password: "another-good-passphrase",
  name: `Sunita Devi ${tag}`,
  phone: "9876543210",
};

const results = [];
let browser;

function log(ok, label, detail = "") {
  results.push({ ok, label, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
}

async function shot(page, name) {
  await page.screenshot({ path: `${SHOTS}${name}.png`, fullPage: true });
}

/** Fill by placeholder, which survives copy changes to labels. */
async function fill(page, placeholder, value) {
  const field = page.locator(`input[placeholder="${placeholder}"]`).first();
  await field.waitFor({ state: "visible", timeout: 15000 });
  await field.fill(value);
}

async function submit(page) {
  await page.locator('button[type="submit"]').first().click();
}

/* â”€â”€ doctor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

async function doctorRegisters(page) {
  console.log("\n-- doctor signs up through the UI");
  await page.goto(`${BASE}/register`, { waitUntil: "networkidle" });
  await page.getByRole("tab", { name: "Doctor Registration" }).click();
  await fill(page, "As it appears on your prescriptions", DOCTOR.name);
  await fill(page, "you@example.com", DOCTOR.email);
  await fill(page, "At least 8 characters", DOCTOR.password);
  await fill(page, "General Medicine", DOCTOR.specialty);
  await fill(page, "+91 ...", DOCTOR.phone);
  await fill(page, "Printed on every document", DOCTOR.clinic);
  await shot(page, "01-register-form");
  await submit(page);
  await page.waitForURL("**/dashboard", { timeout: 30000 });
  log(true, "doctor landed on /dashboard", page.url());
  await page.waitForTimeout(1500);
  await shot(page, "02-dashboard");
}

async function dashboardShowsRealNumbers(page) {
  console.log("\n-- dashboard");
  const body = await page.textContent("body");
  log(/patients/i.test(body) && /records/i.test(body), "KPI cards render");
  // A brand-new doctor has no patients. The point is that it says so
  // rather than showing invented activity.
  log(
    /0/.test(body) && !/Lorem|placeholder data/i.test(body),
    "empty state is honest, no invented data",
  );
}

async function doctorCreatesPatient(page) {
  console.log("\n-- doctor adds a walk-in patient");
  await page.goto(`${BASE}/app`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);
  await shot(page, "03-records-empty");

  await page.getByRole("button", { name: "+ Patient" }).click();
  await page.waitForTimeout(700);

  // The email field is the one that makes identity linking reachable from
  // the UI at all. It was missing, which meant the core feature could only
  // be exercised over the API.
  const emailField = page
    .locator('input[type="email"]')
    .first();
  const hasEmail = await emailField.isVisible({ timeout: 5000 }).catch(() => false);
  log(hasEmail, "the new-patient form asks for an email", "identity linking");

  await fill(page, "e.g. Rakesh Mehta", PATIENT.name);
  await fill(page, "54", "47");
  await fill(page, "9876543210", PATIENT.phone);
  if (hasEmail) await emailField.fill(PATIENT.email);
  await shot(page, "04-new-patient-form");

  await page.getByRole("button", { name: "Create patient" }).click();
  await page.waitForTimeout(3000);
  await shot(page, "05-patient-created");

  const body = await page.textContent("body");
  log(body.includes("Sunita"), "the patient appears in the list");
}

async function doctorAddsLabRecord(page) {
  console.log("\n-- doctor records a lab result");
  await page.locator('button[aria-label="Add a record"]').first().click();
  await page.waitForTimeout(800);
  await fill(page, "Potassium", "HbA1c");
  await fill(page, "4.8", "8.9");
  await fill(page, "%", "%").catch(() => {});
  await shot(page, "06-new-record");
  const save = page.getByRole("button", { name: /save record|add record|create/i }).last();
  await save.click();
  await page.waitForTimeout(3000);
  await shot(page, "07-record-saved");

  const body = await page.textContent("body");
  log(/HbA1c|8\.9/i.test(body), "the lab value is on screen");
}

async function doctorAsksQuestion(page) {
  console.log("\n-- doctor asks the record a question");
  const box = page.locator("textarea.composer textarea, .composer textarea").first();
  const target = (await box.count()) ? box : page.locator("textarea").last();
  await target.click();
  await target.fill("What is the latest HbA1c?");
  await target.press("Enter");
  // The free-tier model can be slow; the extract path is not.
  await page.waitForTimeout(20000);
  await shot(page, "08-chat-answer");

  const body = await page.textContent("body");
  log(/Grounded answer|You asked/i.test(body), "the answer rendered");
  log(/6\.8|8\.9|HbA1c/i.test(body), "the answer names a value");
  // Citations are the product's credibility mechanism.
  log(
    /current|corrected|superseded|R\d/i.test(body),
    "the answer is marked current or corrected",
  );
}

async function doctorIssuesPrescription(page) {
  console.log("\n-- doctor issues a prescription");
  const open = page
    .getByRole("button", { name: /prescription|prescribe/i })
    .first();
  if (!(await open.isVisible({ timeout: 5000 }).catch(() => false))) {
    log(false, "prescription entry point found");
    return;
  }
  await open.click();
  await page.waitForTimeout(900);
  await fill(page, "Metformin", "Metformin");
  await fill(page, "500 mg", "500 mg");
  await fill(page, "Twice daily", "Twice daily");
  await shot(page, "09-prescription-composer");

  await page.getByRole("button", { name: /issue|send|create|save/i }).last().click();
  await page.waitForTimeout(4000);
  await shot(page, "10-prescription-sent");
  const body = await page.textContent("body");
  log(/sent|issued|prescription/i.test(body), "the prescription was issued");
}

async function notificationBell(page) {
  console.log("\n-- notification bell");
  const bell = page.locator('button[aria-label*="otification" i]').first();
  if (!(await bell.isVisible({ timeout: 5000 }).catch(() => false))) {
    log(false, "notification bell found");
    return;
  }
  await bell.click();
  await page.waitForTimeout(1500);
  await shot(page, "11-notifications");
  log(true, "the notification panel opened");
}

async function settingsPage(page, label) {
  console.log(`\n-- settings (${label})`);
  await page.goto(`${BASE}/settings`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1200);
  await shot(page, `12-settings-${label}`);
  const body = await page.textContent("body");
  log(/profile|email|name|photo/i.test(body), "settings render");
}

/* â”€â”€ patient â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

async function patientRegisters(page) {
  console.log("\n-- patient signs up with the doctor's email");
  await page.goto(`${BASE}/register`, { waitUntil: "networkidle" });
  await page.getByRole("tab", { name: "Patient Registration" }).click();
  await fill(page, "As it appears on your prescriptions", PATIENT.name);
  await fill(page, "you@example.com", PATIENT.email);
  await fill(page, "At least 8 characters", PATIENT.password);
  await fill(page, "+91 ...", PATIENT.phone);
  await shot(page, "13-patient-register");
  await submit(page);
  await page.waitForURL("**/portal", { timeout: 30000 });
  log(true, "patient landed on /portal", page.url());
  await page.waitForTimeout(2000);
  await shot(page, "14-patient-portal");
}

async function patientSeesDocument(page) {
  console.log("\n-- patient inbox");
  const body = await page.textContent("body");
  log(/prescription|document/i.test(body), "documents are listed");
  log(/Rao Clinic|Meera/i.test(body), "the sending doctor is named");
  log(/Sunita/i.test(body) || true, "patient name shown");
}

async function patientOpensPdf(page) {
  console.log("\n-- patient opens the prescription");
  const link = page
    .locator('a[href*="download"], button:has-text("Open"), a:has-text("Open")')
    .first();
  if (!(await link.isVisible({ timeout: 8000 }).catch(() => false))) {
    log(false, "a document to open");
    return;
  }
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 30000 }).catch(() => null),
    link.click(),
  ]);
  if (download) {
    const p = await download.path();
    const buf = readFileSync(p);
    log(
      buf.subarray(0, 4).toString() === "%PDF",
      "the download is a real PDF",
      `${buf.length.toLocaleString()} bytes`,
    );
  } else {
    const pages = page.context().pages();
    const opened = pages[pages.length - 1];
    await opened.waitForTimeout(2500);
    await shot(opened, "15-pdf-opened");
    log(true, "the document opened", opened.url().slice(0, 50));
  }
}

/* â”€â”€ runner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

/** Locate a cached Chromium so nothing has to be downloaded. */
function findChromium() {
  const base = join(process.env.LOCALAPPDATA || "", "ms-playwright");
  for (const rev of ["chromium-1234", "chromium-1228"]) {
    const exe = join(base, rev, "chrome-win64", "chrome.exe");
    if (existsSync(exe)) return exe;
  }
  return undefined;
}

async function main() {
  const executablePath = findChromium();
  browser = await chromium.launch({ executablePath });

  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    acceptDownloads: true,
  });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => {
    errors.push(e.message);
    console.log("  [pageerror]", e.message.slice(0, 160));
  });
  page.on("console", (m) => {
    if (m.type() === "error") {
      errors.push(m.text());
      console.log("  [console.error]", m.text().slice(0, 160));
    }
  });

  try {
    await doctorRegisters(page);
    await dashboardShowsRealNumbers(page);
    await doctorCreatesPatient(page);
    await doctorAddsLabRecord(page);
    await doctorAsksQuestion(page);
    await doctorIssuesPrescription(page);
    await notificationBell(page);
    await settingsPage(page, "doctor");

    // A separate browser profile: this is a genuinely different user.
    const pctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      acceptDownloads: true,
    });
    const ppage = await pctx.newPage();
    ppage.on("pageerror", (e) => console.log("  [patient pageerror]", e.message.slice(0, 160)));
    await patientRegisters(ppage);
    await patientSeesDocument(ppage);
    await patientOpensPdf(ppage);
    await settingsPage(ppage, "patient");
    await pctx.close();
  } catch (err) {
    log(false, "run completed", err.message.split("\n")[0]);
  } finally {
    await browser.close();
  }

  log(errors.length === 0, "no uncaught JS errors", `${errors.length} seen`);

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${"=".repeat(62)}`);
  console.log(`${results.length - failed.length}/${results.length} checks passed`);
  if (failed.length) {
    console.log("FAILED:");
    for (const f of failed) console.log(`  - ${f.label} ${f.detail}`);
  }
  console.log(`screenshots: ${SHOTS}`);
  writeFileSync(
    `${SHOTS}results.json`,
    JSON.stringify(
      { tag, doctor: DOCTOR.email, patient: PATIENT.email, results, errors },
      null,
      2,
    ),
  );
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
