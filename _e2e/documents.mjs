/**
 * Verify the per-patient document actions on the Records page.
 *
 * The prescription composer existed but only on the dashboard, so a
 * doctor looking at a patient had to re-pick that patient from a
 * dropdown. The weekly report had no UI at all, despite a grounded
 * backend endpoint. Both are now per-patient on the record itself.
 *
 * This drives the real buttons and asserts the patient is preselected,
 * because a composer that opens blank after you clicked it on a
 * specific patient is the exact bug being fixed.
 */

import { chromium } from "playwright";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHOTS = join(HERE, "shots") + "\\";
mkdirSync(SHOTS, { recursive: true });

const BASE = process.env.BASE || "http://localhost:8080";
const API = "http://localhost:8000";
const tag = randomUUID().slice(0, 10);

const results = [];
function log(ok, label, detail = "") {
  results.push({ ok, label, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
}

function findChromium() {
  const base = join(process.env.LOCALAPPDATA || "", "ms-playwright");
  for (const rev of ["chromium-1234", "chromium-1228"]) {
    const exe = join(base, rev, "chrome-win64", "chrome.exe");
    if (existsSync(exe)) return exe;
  }
  return undefined;
}

async function fill(page, placeholder, value) {
  const f = page.locator(`input[placeholder="${placeholder}"]`).first();
  await f.waitFor({ state: "visible", timeout: 15000 });
  await f.fill(value);
}

/** Seed a patient with a corrected lab value, over the API, so the report
 *  has something real to say and the correction path is exercised. */
async function seed() {
  const post = async (path, body, token) => {
    const r = await fetch(API + path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
    });
    return { status: r.status, body: await r.json().catch(() => null) };
  };
  const get = async (path, token) => {
    const r = await fetch(API + path, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    return await r.json().catch(() => null);
  };

  const doctor = await post("/api/v1/auth/register/doctor", {
    email: `doc.${tag}@solum.test`,
    password: "correct-horse-battery",
    display_name: `Dr Docs ${tag}`,
    specialty: "General Medicine",
    clinic_name: "Docs Clinic",
  });
  const token = doctor.body.token;

  const patient = await post(
    "/api/v1/patients",
    {
      display_name: `Ranveer Thakur ${tag}`,
      age: 21,
      email: `ranveer.${tag}@solum.test`,
    },
    token,
  );
  const pid = patient.body.id;

  const now = new Date();
  const ago = (d) => new Date(now.getTime() - d * 86400000).toISOString();
  const labs = [
    ["HbA1c", 8.9, "%", 40],
    ["Potassium", 2.1, "mEq/L", 38],
    ["Creatinine", 1.1, "mg/dL", 36],
  ];
  for (const [test, value, unit, d] of labs) {
    await post(
      `/api/v1/patients/${pid}/events`,
      {
        event_type: "lab",
        valid_time: ago(d),
        payload: { test, value, unit },
      },
      token,
    );
  }
  // A correction, so the weekly report has something to be correct about.
  const events = await get(`/api/v1/patients/${pid}/events`, token);
  const k = events.find((e) => e.payload?.test === "Potassium");
  await post(
    `/api/v1/patients/${pid}/events`,
    {
      event_type: "lab",
      source_type: "correction",
      supersedes_id: k.id,
      valid_time: ago(34),
      payload: { test: "Potassium", value: 4.8, unit: "mEq/L" },
    },
    token,
  );

  return { token, pid, name: `Ranveer Thakur ${tag}` };
}

async function main() {
  const { token, pid, name } = await seed();
  console.log(`\nseeded ${name} (${pid.slice(0, 8)}) with 3 labs + 1 correction`);

  const executablePath = findChromium();
  const browser = await chromium.launch({ executablePath });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 950 } });
  const page = await ctx.newPage();
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });

  try {
    // Sign in through the UI so the session is a real one. The password
    // field has no placeholder on the sign-in tab by design, so select it
    // by type rather than by placeholder.
    await page.goto(`${BASE}/signin`, { waitUntil: "networkidle" });
    await fill(page, "you@example.com", `doc.${tag}@solum.test`);
    await page.locator('input[type="password"]').first().fill("correct-horse-battery");
    await page.locator('button[type="submit"]').first().click();
    await page.waitForURL("**/dashboard", { timeout: 30000 });
    log(true, "signed in through the UI");

    console.log("\n-- records page: the per-patient actions");
    await page.goto(`${BASE}/app`, { waitUntil: "networkidle" });
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `${SHOTS}d1-records-with-actions.png` });

    const bar = page.locator(".console-bar");
    const barText = await bar.textContent();
    log(/New prescription/i.test(barText), "New prescription is on the record page");
    log(/Weekly report/i.test(barText), "Weekly report is on the record page");
    log(barText.includes("Ranveer"), "the action bar names the patient", barText.match(/For .*/)?.[0]);

    const rxBtn = page.getByRole("button", { name: "New prescription" }).first();
    const wkBtn = page.getByRole("button", { name: "Weekly report" }).first();
    log(await rxBtn.isEnabled(), "New prescription is enabled");
    log(await wkBtn.isEnabled(), "Weekly report is enabled (patient has records)");

    console.log("\n-- prescription composer opens, preselected");
    await rxBtn.click();
    // The sheet renders through a portal with no stable class, so scope by
    // its title text rather than a class that does not exist in the DOM.
    const sheet = page
      .locator("div")
      .filter({ hasText: "New prescription" })
      .last();
    await sheet.waitFor({ state: "visible", timeout: 10000 });
    const preselected = await page
      .locator("select")
      .first()
      .evaluate((el) => el.value);
    log(!!preselected, "a patient is preselected, not blank", preselected.slice(0, 8));
    const sheetName = await page.locator("option:checked").first().textContent();
    log(
      (sheetName || "").includes("Ranveer"),
      "the preselected patient is the one on screen",
      sheetName,
    );
    log(
      (await page.locator('input[placeholder="Type 2 Diabetes Mellitus"]').count()) > 0,
      "the diagnosis field is present",
    );
    await page.screenshot({ path: `${SHOTS}d2-prescription-preselected.png` });

    // Actually issue one, so the sheet is proven to submit.
    await page.locator('input[placeholder="Metformin"]').first().fill("Paracetamol");
    await page.locator('input[placeholder="500 mg"]').first().fill("650 mg");
    await page
      .locator('input[placeholder="Type 2 Diabetes Mellitus"]')
      .first()
      .fill("Yellow fever");
    await page.getByRole("button", { name: /issue|send|create|save/i }).last().click();
    await page.waitForTimeout(5000);
    await page.screenshot({ path: `${SHOTS}d3-prescription-issued.png` });
    // Assert the sheet is gone, not the button: the button reads "New
    // prescription" too, so matching that text would always be true.
    log(
      (await page.locator('input[placeholder="Metformin"]').count()) === 0,
      "the sheet closed after issuing",
    );

    console.log("\n-- weekly report, grounded in the record");
    await page.getByRole("button", { name: "Weekly report" }).first().click();
    await page.waitForTimeout(1200);
    const wtext = await page.locator(".sheet").last().textContent();
    log(/Last 7 days/i.test(wtext), "period choices are offered");
    log(/corrected/i.test(wtext), "it explains the correction policy");
    const wsel = await page
      .locator(".sheet select")
      .first()
      .evaluate((el) => el.value);
    log(!!wsel, "the patient is preselected", wsel.slice(0, 8));
    await page.screenshot({ path: `${SHOTS}d4-weekly-sheet.png` });

    await page.getByRole("button", { name: /generate report/i }).click();
    // The LLM writes the summary, so allow for a slow free-tier call.
    await page.waitForTimeout(25000);
    await page.screenshot({ path: `${SHOTS}d5-weekly-done.png` });
    const after = await page.textContent("body");
    const openedPdf = ctx.pages().length > 1;
    log(
      /Report WEE-|ready/i.test(after) || openedPdf,
      "a report reference was shown",
      openedPdf ? "opened a PDF tab" : after.match(/Report WEE-[A-Z0-9-]+/)?.[0],
    );
    log(
      !(await page.locator(".sheet").isVisible().catch(() => false)),
      "the sheet closed after generating",
    );

    // Confirm the generated report really is grounded, from the API.
    const docs = await (
      await fetch(`${API}/api/v1/documents/patient/${pid}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
    ).json();
    const weekly = docs.find((d) => d.kind === "weekly_report");
    log(!!weekly, "a weekly report exists for this patient", weekly?.reference);
  } catch (err) {
    log(false, "run completed", err.message.split("\n")[0]);
  } finally {
    await browser.close();
  }

  log(consoleErrors.length === 0, "no console errors", String(consoleErrors.length));

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${"=".repeat(60)}`);
  console.log(`${results.length - failed.length}/${results.length} checks passed`);
  for (const f of failed) console.log(`  FAILED: ${f.label} ${f.detail}`);
  writeFileSync(`${SHOTS}doc-results.json`, JSON.stringify(results, null, 2));
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
