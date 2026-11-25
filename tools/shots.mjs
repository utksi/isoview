// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Screenshots of fixed scenes, for checking the look of a build (run by .github/workflows/pages.yml on GitHub's
 * runner, against the assembled site served locally). Each shot uses a fresh browser context, so no saved scene or
 * setting carries over; the theme and (on the test site) the typeface are put into localStorage before the page loads.
 * Rendering is WebGL2 on SwiftShader (software), so the WebGPU path and real frame rates are not shown.
 *
 * Usage: node tools/shots.mjs <chrome executable> <out dir> <label>=<site url> [<label>=<site url> ...]
 * Writes <out dir>/<label>-<scene>-<theme>[-<font>].png at 1440 x 900 CSS px, device pixel ratio 1 (one scene also 2).
 */
import { mkdirSync } from "node:fs";
import puppeteer from "puppeteer-core";

const [chrome, out, ...sites] = process.argv.slice(2);
// scene: a sample from the opening page (data-example), or null for the opening page itself
const SCENES = [["home", null], ["rutile", "rutile"], ["density", "si"], ["slice2d", "sislice"], ["orbital", "n2"], ["md", "vasprun"]];
const FONTS = ["system", "inter", "source", "plex"];

mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: chrome, headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--hide-scrollbars"],
});

/** One screenshot (a failure is reported and skipped): open `url` with theme and font preset, open a sample, wait until it is drawn, save the PNG. */
async function shot(url, file, opts) {
  try { await shot1(url, file, opts); } catch (e) { console.log(file, "FAILED:", e.message); }
}
async function shot1(url, file, { theme, example, font, dpr = 1 }) {
  const ctx = await browser.createBrowserContext();
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(String(e)));
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: dpr });
  await page.evaluateOnNewDocument((theme, font) => {
    for (const p of ["isoview-", "isoview-next-"]) {
      localStorage.setItem(p + "theme", theme);
      if (font) localStorage.setItem(p + "font", font);
    }
  }, theme, font || "");
  await page.goto(url + (url.includes("?") ? "&" : "?") + "gl=webgl2", { waitUntil: "networkidle0", timeout: 60000 });
  if (example) {
    await page.click(`[data-example="${example}"]`);
    await page.waitForFunction(() => document.getElementById("empty").hidden && document.getElementById("busy").hidden, { timeout: 90000 });
  }
  await new Promise(r => setTimeout(r, 2500)); // WHY: meshing, slice images and fonts settle after the busy flag clears
  await page.screenshot({ path: file });
  await ctx.close();
  if (errors.length) console.log(file, "page errors:", errors.join(" | "));
  else console.log(file);
}

for (const s of sites) {
  const [label, url] = s.split("=");
  for (const [name, example] of SCENES) for (const theme of ["light", "dark"]) {
    await shot(url, `${out}/${label}-${name}-${theme}.png`, { theme, example });
  }
  await shot(url, `${out}/${label}-rutile-light-dpr2.png`, { theme: "light", example: "rutile", dpr: 2 });
  if (label === "next") for (const font of FONTS) await shot(url, `${out}/${label}-rutile-light-${font}.png`, { theme: "light", example: "rutile", font });
}
await browser.close();
