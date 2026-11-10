/**
 * Tabulate the Gardner–Mindlin normal form's limit cycle over (α, β): period T (in τ), and the
 * X range (for mapping labial displacement to valve opening). Output: crates/chimera/src/sources/syrinxTable.generated.json
 */
import { writeFileSync } from "node:fs";
const rhs = (x: number, y: number, a: number, b: number): [number, number] => [y, -a - b * x - x * x * x - x * x * y + x * x - x * y];
function cycle(a: number, b: number) {
  let x = 0.3, y = 0;
  const h = 0.002;
  const N = 300000;
  let mn = Infinity, mx = -Infinity, sum = 0, cnt = 0;
  const cross: number[] = [];
  let prev = x;
  for (let i = 0; i < N; i++) {
    const [k1x, k1y] = rhs(x, y, a, b);
    const [k2x, k2y] = rhs(x + 0.5 * h * k1x, y + 0.5 * h * k1y, a, b);
    const [k3x, k3y] = rhs(x + 0.5 * h * k2x, y + 0.5 * h * k2y, a, b);
    const [k4x, k4y] = rhs(x + h * k3x, y + h * k3y, a, b);
    x += (h / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    y += (h / 6) * (k1y + 2 * k2y + 2 * k3y + k4y);
    if (i > N / 2) {
      mn = Math.min(mn, x); mx = Math.max(mx, x); sum += x; cnt++;
    }
    if (i > N / 2 && prev < 0 && x >= 0) cross.push(i);
    prev = x;
  }
  const mean = sum / cnt;
  const T = cross.length > 2 ? ((cross[cross.length - 1] - cross[0]) / (cross.length - 1)) * h : NaN;
  return { T, min: mn, max: mx, mean };
}
const alphas = Array.from({ length: 13 }, (_, i) => 0.07 + (i / 12) * 0.53);
const betas = Array.from({ length: 13 }, (_, i) => 0.1 + (i / 12) * 1.2);
const T: number[][] = [], lo: number[][] = [], hi: number[][] = [];
for (const b of betas) {
  const rT: number[] = [], rl: number[] = [], rh: number[] = [];
  for (const a of alphas) {
    const c = cycle(a, b);
    rT.push(Math.round(c.T * 1000) / 1000);
    rl.push(Math.round(c.min * 1000) / 1000);
    rh.push(Math.round(c.max * 1000) / 1000);
  }
  T.push(rT); lo.push(rl); hi.push(rh);
}
writeFileSync("crates/chimera/src/sources/syrinxTable.generated.json", JSON.stringify({ alphas, betas, T, lo, hi }));
console.log("T range", Math.min(...T.flat().filter(Number.isFinite)), Math.max(...T.flat()), "NaN cells", T.flat().filter((v) => !Number.isFinite(v)).length);
