/**
 * UI-gap regression: correct the FIRST record from the Console timeline.
 *
 * The timeline used to only let a user "correct this record again" for an
 * event already in a correction pair; there was no affordance to create
 * the first one from a loose row. Acceptance:
 *   1. open patient Console page
 *   2. add one lab record
 *   3. click the loose record's "correct" button
 *   4. enter corrected value + why
 *   5. press Save correction
 *   6. expect toast "Correction recorded"
 *   7. expect the superseded pair visible and chat grounded to corrected value
 *
 * Run: node _e2e/ui_correct.mjs
 */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { existsSync } from "node:fs";
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
  name: `Dr First Correction ${tag}`,
  specialty: "General Medicine",
  phone: "+910000",
};
const PATIENT = { email: `patient.${tag}@solum.test`, name: `Ravi Kumar ${tag}`, phone: "8000000000" };

const results = [];
function log(ok, label, detail = "") {
  results.push({ ok, label, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
}

async function fill(page, placeholder, value) {
  const f = page.locator(`input[placeholder="${placeholder}"]`).first();
  await f.waitFor({ state: "visible", timeout: 15000 });
  await f.fill(value);
}

function findChromium() {
  const base = join(process.env.LOCALAPPDATA || "", "ms-playwright");
  const candidates = [
    join(base, "chromium-1234/chrome-win64/chrome.exe"),
    join(base, "chromium-1228/chrome-win64/chrome.exe"),
    join(base, "chromium-1217/chrome-headless-shell-win64/chrome-headless-shell.exe"),
  ];
  for (const c of candidates) { if (existsSync(c)) return c; }
}

async function main() {
  const browser = await chromium.launch({ executablePath: findChromium() });
  const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });

  await page.goto(`${BASE}/register`, { waitUntil: "networkidle" });
  await page.getByRole("tab", { name: "Doctor Registration" }).click();
  await fill(page, "As it appears on your prescriptions", DOCTOR.name);
  await fill(page, "you@example.com", DOCTOR.email);
  await fill(page, "At least 8 characters", DOCTOR.password);
  await fill(page, "General Medicine", DOCTOR.specialty);
  await page.locator('button[type="submit"]').first().click();
  try { await page.waitForURL("**/dashboard", { timeout: 30000 }); log(true, "doctor registered"); } catch { log(false, "doctor registered"); process.exit(1); }

  await page.goto(`${BASE}/app`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "+ Patient" }).click();
  await fill(page, "e.g. Rakesh Mehta", PATIENT.name);
  await fill(page, "54", "48");
  await fill(page, "9876543210", PATIENT.phone);
  const emailField = page.locator('input[type="email"]').first();
  if (await emailField.isVisible({ timeout: 5000 }).catch(() => false)) {
    await emailField.fill(PATIENT.email);
  }
  await page.getByRole("button", { name: "Create patient" }).click();
  await page.waitForTimeout(2500);
  log((await page.textContent("body")).includes(PATIENT.name), "patient created");

  // Add one lab (loose) so there is something to correct.
  await page.locator('button[aria-label="Add a record"]').first().click();
  await fill(page, "Potassium", "HbA1c");
  await fill(page, "4.8", "8.9");
  await fill(page, "%", "%").catch(() => {});
  await page.getByRole("button", { name: /save record|add record|create/i }).last().click();
  await page.waitForTimeout(3000);
  log(/HbA1c|8\.9/i.test(await page.textContent("body")), "original record present");
  await page.screenshot({ path: `${SHOTS}correct-before.png`, fullPage: true });

  // The UI gap fix: a loose record should now expose a correct affordance.
  const looseCorrectBtn = page.locator('.ev button:has-text("correct")').first();
  const visible = await looseCorrectBtn.isVisible({ timeout: 10000 }).catch(() => false);
  log(visible, "loose record has a correct button");
  if (!visible) { await page.screenshot({ path: `${SHOTS}correct-missing.png`, fullPage: true }); await browser.close(); process.exit(1); }

  await looseCorrectBtn.click();
  await page.waitForTimeout(1000);
  await fill(page, "4.8", "7.0");
  const amendment = page.locator('textarea').last();
  if (await amendment.isVisible({ timeout: 5000 }).catch(() => false)) {
    await amendment.fill("Repeat lab on 2026-09-20, original transcription error");
  }
  await page.screenshot({ path: `${SHOTS}correct-form.png`, fullPage: true });
  await page.getByRole("button", { name: /Save correction/i }).last().click();
  await page.waitForTimeout(5000);

  const toast = await page.textContent("body");
  log(/Correction recorded/i.test(toast), "correction toast shown");
  await page.screenshot({ path: `${SHOTS}correct-done.png`, fullPage: true });

  // Timeline pairs for the patient should now expose the superseded row.
  const body = await page.textContent("body");
  log(/superseded|corrected/i.test(body), "timeline shows superseded/correction");

  // Chat: ask a present-tense question and verify grounded to 7.0%.
  const ta = page.locator("textarea").last();
  await ta.click();
  await ta.fill("What is the latest HbA1c?");
  await ta.press("Enter");
  await page.waitForTimeout(20000);
  const answer = (await page.locator(".msg").last().textContent()) || "";
  const hasCorrected = /7(?:[.,]0)?/i.test(answer);
  const hasOldAsLatest = /Latest.*8\.9/i.test(answer);
  log(hasCorrected && !hasOldAsLatest, "chat answers with the corrected value", answer.replace(/\s+/g, " ").slice(0, 180));
  await page.screenshot({ path: `${SHOTS}correct-chat.png`, fullPage: true });

  await browser.close();
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  process.exit(failed.length);
}

main().catch((e) => { console.error(String(e)); process.exit(1); });
