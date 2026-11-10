/**
 * Build the compact pronunciation dictionary shipped to the browser.
 * Usage: bun tools/build-dict.ts <path/to/cmudict.dict> <path/to/LICENSE>
 * Output: public/data/cmudict.txt.gz  (lines: "word<TAB>PHONES", first pronunciation only)
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { gzipSync } from "node:zlib";

const [dictPath, licensePath] = process.argv.slice(2);
if (!dictPath || !licensePath) throw new Error("usage: bun tools/build-dict.ts <cmudict.dict> <LICENSE>");
const lines = readFileSync(dictPath, "utf8").split("\n");
const out: string[] = [];
const seen = new Set<string>();
for (const raw of lines) {
  const line = raw.split("#")[0].trim();
  if (!line) continue;
  const sp = line.indexOf(" ");
  if (sp < 0) continue;
  let word = line.slice(0, sp).toLowerCase();
  const variant = word.match(/\((\d+)\)$/);
  if (variant) continue; // keep first pronunciation only
  if (!/^[a-z'][a-z'.\-]*$/.test(word) || seen.has(word)) continue;
  seen.add(word);
  const phones = line.slice(sp + 1).trim().split(/\s+/).join(" ");
  out.push(`${word}\t${phones}`);
}
mkdirSync("public/data", { recursive: true });
const gz = gzipSync(Buffer.from(out.join("\n"), "utf8"), { level: 9 });
writeFileSync("public/data/cmudict.txt.gz", gz);
writeFileSync("public/data/CMUDICT-LICENSE.txt", readFileSync(licensePath, "utf8"));
console.log(`${out.length} words, ${(gz.length / 1024).toFixed(0)} KiB gzipped`);
