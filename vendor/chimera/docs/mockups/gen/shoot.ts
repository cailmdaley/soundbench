/** Screenshot the mockup: bun shoot.ts <html> <out-prefix> [dpr] [clips as name:x,y,w,h ...] */
import { chromium } from "playwright-core";

const [file, out, dprArg, ...clips] = process.argv.slice(2);
const dpr = Number(dprArg ?? 1);
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: dpr });
const errors: string[] = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(e.message));
await page.goto("file://" + file, { waitUntil: "load" });
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(250);
await page.screenshot({ path: `${out}-full.png` });
for (const c of clips) {
  const [name, rect] = c.split(":");
  const [x, y, width, height] = rect.split(",").map(Number);
  await page.screenshot({ path: `${out}-${name}.png`, clip: { x, y, width, height } });
}
const fontsOk = await page.evaluate(() => document.fonts.check('300 12px "Hershey"') && document.fonts.check('500 12px "Hershey"'));
console.log("fonts", fontsOk, "errors", errors.length ? errors.join(" | ") : "none");
await browser.close();
