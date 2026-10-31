/**
 * Verify the profile picture actually appears, in a real browser.
 *
 * The previous check curled `GET /api/v1/documents/photo` and got a 200,
 * so the photo upload was declared working. It was not. The `<img>` tag
 * cannot send an `Authorization` header, so the browser got a 401, the
 * `onError` handler hid the image, and the avatar rendered as blank
 * space. The upload had succeeded the whole time; only the display was
 * broken, and the failure was silent.
 *
 * So this test does what a user does: selects a real file through the
 * real file input, and asserts that a rendered <img> with non-zero size
 * appears. It also asserts the fallback is a visible element rather than
 * a hole, which is the part the old code got wrong.
 */

import { chromium } from "playwright";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { deflateSync } from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHOTS = join(HERE, "shots") + "\\";
mkdirSync(SHOTS, { recursive: true });

const BASE = process.env.BASE || "http://localhost:8080";
const tag = randomUUID().slice(0, 10);

const results = [];
function log(ok, label, detail = "") {
  results.push({ ok, label, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
}

/** A real 64x64 PNG with a visible magenta square, so a rendered image is
 *  distinguishable from a broken-image box or an empty div. */
function makePng() {
  const W = 64;
  const crcTable = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[n] = c >>> 0;
  }
  const crc32 = (buf) => {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body));
    return Buffer.concat([len, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(W, 0);
  ihdr.writeUInt32BE(W, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // truecolour
  const raw = [];
  for (let y = 0; y < W; y++) {
    const row = Buffer.alloc(1 + W * 3);
    for (let x = 0; x < W; x++) {
      const i = 1 + x * 3;
      row[i] = 255; // R
      row[i + 1] = 0; // G
      row[i + 2] = 200; // B  -> magenta
    }
    raw.push(row);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(Buffer.concat(raw))),
    chunk("IEND", Buffer.alloc(0)),
  ]);
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

async function main() {
  const executablePath = findChromium();
  const browser = await chromium.launch({ executablePath });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();

  // Any request that 401s or 403s while we are on Settings is the bug.
  const authFailures = [];
  page.on("response", (r) => {
    if (r.url().includes("/api/") && (r.status() === 401 || r.status() === 403)) {
      authFailures.push(`${r.status()} ${r.url()}`);
    }
  });
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });

  try {
    console.log("\n-- register a doctor");
    await page.goto(`${BASE}/register`, { waitUntil: "networkidle" });
    await page.getByRole("tab", { name: "Doctor Registration" }).click();
    await fill(page, "As it appears on your prescriptions", `Dr Photo ${tag}`);
    await fill(page, "you@example.com", `photo.${tag}@solum.test`);
    await fill(page, "At least 8 characters", "correct-horse-battery");
    await fill(page, "General Medicine", "General Medicine");
    await fill(page, "+91 ...", "+919000000001");
    await page.locator('button[type="submit"]').first().click();
    await page.waitForURL("**/dashboard", { timeout: 30000 });
    log(true, "doctor registered");

    console.log("\n-- settings, before any photo: initials must be visible");
    await page.goto(`${BASE}/settings`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);
    let avatar = page.locator(".set-avatar").first();
    log(await avatar.isVisible(), "an avatar element is visible");
    let box = await avatar.boundingBox();
    log(box && box.width > 20, "the avatar has real size", box ? `${Math.round(box.width)}x${Math.round(box.height)}` : "no box");
    await page.screenshot({ path: `${SHOTS}p1-settings-before.png` });

    console.log("\n-- select a real file through the real input");
    const png = makePng();
    await page.setInputFiles('input[type="file"]', {
      name: "avatar.png",
      mimeType: "image/png",
      buffer: png,
    });

    // Give the upload, the presigned-url fetch, and the image decode.
    await page.waitForTimeout(4000);
    await page.screenshot({ path: `${SHOTS}p2-settings-after.png` });

    const img = page.locator("img.set-avatar").first();
    const hasImg = await img.count();
    if (hasImg) {
      log(true, "an <img> is rendered, not the initials fallback");
      const ibox = await img.boundingBox();
      log(ibox && ibox.width > 20, "the image has real size", ibox ? `${Math.round(ibox.width)}x${Math.round(ibox.height)}` : "no box");

      // The real assertion: the browser actually decoded the bytes.
      const natural = await img.evaluate((el) => ({
        w: el.naturalWidth,
        h: el.naturalHeight,
        complete: el.complete,
        src: el.currentSrc || el.src,
      }));
      log(natural.complete && natural.w > 0, "the browser decoded the image", `${natural.w}x${natural.h}`);
      log(
        !natural.src.includes("/api/v1/documents/photo"),
        "the src is a presigned object-storage URL, not the API",
        natural.src.slice(0, 58),
      );
    } else {
      log(false, "an <img> is rendered after upload", "still showing initials");
    }

    log(authFailures.length === 0, "no 401/403 from the API", authFailures.join(", ") || "clean");

    console.log("\n-- survives a page reload (not just in-memory state)");
    await page.reload({ waitUntil: "networkidle" });
    await page.waitForTimeout(2500);
    const afterReload = await page.locator("img.set-avatar").count();
    log(afterReload > 0, "the photo is still there after a reload");
    const rb = afterReload
      ? await page.locator("img.set-avatar").first().evaluate((el) => el.naturalWidth)
      : 0;
    log(rb > 0, "and it still decodes after a reload", `${rb}px natural`);
    await page.screenshot({ path: `${SHOTS}p3-settings-reload.png` });
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
  writeFileSync(`${SHOTS}photo-results.json`, JSON.stringify(results, null, 2));
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
