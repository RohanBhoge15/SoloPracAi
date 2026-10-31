/**
 * Complete a crop end to end, in a real browser.
 *
 * The first version of the cropper shipped broken: `onMediaLoaded` reports
 * a `MediaSize` plain object, not a drawable image, so `ctx.drawImage`
 * threw and every crop failed with "Could not crop that image". It
 * compiled, it rendered, the frame looked right - and it was broken.
 *
 * So this asserts the outcome, not the presence of the dialog: the file is
 * selected, the dialog opens, "Use this picture" is pressed, the dialog
 * closes, and the avatar in settings becomes a decoded <img>. A cropper
 * that renders but cannot produce a blob fails here.
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

/** A non-square image with four coloured quadrants, so a crop that is not
 *  actually centred is visible rather than guessed at. */
function makePng(W = 300, H = 200) {
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
  ihdr.writeUInt32BE(H, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const raw = [];
  for (let y = 0; y < H; y++) {
    const row = Buffer.alloc(1 + W * 3);
    for (let x = 0; x < W; x++) {
      const i = 1 + x * 3;
      const left = x < W / 2;
      const top = y < H / 2;
      row[i] = left ? (top ? 255 : 0) : top ? 0 : 255;
      row[i + 1] = left ? 0 : 255;
      row[i + 2] = left ? (top ? 0 : 160) : top ? 200 : 40;
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
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  try {
    await page.goto(`${BASE}/register`, { waitUntil: "networkidle" });
    await page.getByRole("tab", { name: "Doctor Registration" }).click();
    await fill(page, "As it appears on your prescriptions", `Dr Crop ${tag}`);
    await fill(page, "you@example.com", `crop.${tag}@solum.test`);
    await fill(page, "At least 8 characters", "correct-horse-battery");
    await page.locator('button[type="submit"]').first().click();
    await page.waitForURL("**/dashboard", { timeout: 30000 });
    log(true, "registered");

    await page.goto(`${BASE}/settings`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);

    console.log("\n-- select a file: the crop dialog must open");
    await page.setInputFiles('input[type="file"]', {
      name: "me.png",
      mimeType: "image/png",
      buffer: makePng(),
    });
    const dialog = page.locator(".cropper-overlay");
    await dialog.waitFor({ state: "visible", timeout: 10000 });
    log(true, "the crop dialog opened");
    log(
      (await dialog.textContent()).includes("Drag to reposition"),
      "it explains what to do",
    );
    await page.screenshot({ path: `${SHOTS}c1-crop-dialog.png` });

    console.log("\n-- move the zoom slider, so a real crop is exercised");
    const slider = dialog.locator('input[type="range"]');
    await slider.fill("1.6");
    await page.waitForTimeout(700);
    log(true, "zoom applied");

    console.log("\n-- confirm: the dialog must close and the photo must land");
    const confirm = page.getByRole("button", { name: /use this picture/i });
    log(await confirm.isEnabled(), "the confirm button is enabled");
    await confirm.click();
    await page.waitForTimeout(6000);
    await page.screenshot({ path: `${SHOTS}c2-after-crop.png` });

    const stillOpen = await dialog.isVisible().catch(() => false);
    log(!stillOpen, "the crop dialog closed");

    const body = await page.textContent("body");
    log(
      !body.includes("Could not crop that image"),
      'no "Could not crop" error is shown',
    );

    const img = page.locator("img.set-avatar");
    if (await img.count()) {
      const nat = await img.first().evaluate((el) => el.naturalWidth);
      log(nat > 0, "the cropped photo decoded in the browser", `${nat}px natural`);
      const src = await img.first().evaluate((el) => el.currentSrc || el.src);
      log(
        !src.includes("/api/v1/documents/photo"),
        "loaded from object storage, not the API",
        src.slice(0, 46),
      );
    } else {
      log(false, "a cropped photo is displayed", "still showing initials");
    }

    // Prove the crop is square: the stored file must be square too.
    const dims = await page.evaluate(async () => {
      const el = document.querySelector("img.set-avatar");
      if (!el) return null;
      return { w: el.naturalWidth, h: el.naturalHeight };
    });
    if (dims) {
      log(
        dims.w === dims.h,
        "the stored crop is square",
        `${dims.w}x${dims.h}`,
      );
    }
  } catch (err) {
    log(false, "run completed", err.message.split("\n")[0]);
  } finally {
    await browser.close();
  }

  log(errors.length === 0, "no console/page errors", String(errors.length));

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${"=".repeat(60)}`);
  console.log(`${results.length - failed.length}/${results.length} checks passed`);
  for (const f of failed) console.log(`  FAILED: ${f.label} ${f.detail}`);
  writeFileSync(`${SHOTS}crop-results.json`, JSON.stringify(results, null, 2));
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
