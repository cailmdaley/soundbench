/**
 * Petting a head: wiggle the pointer over the top of it for a moment and it closes its eyes, and
 * heart outlines float up beside it, as if it likes it. The view feeds the pointer's path over the
 * crown (in cm on the head, so it doesn't matter how big the head is drawn) and draws the hearts.
 */
import { V } from "../colors.ts";

/** path over the crown (cm) that makes a head happy from rest, less what fades meanwhile: an
 *  ordinary wiggle (30 cm/s on the head: a mouse at ~700 px/s over a desktop head) gets there in
 *  about a second and a half */
const HAPPY_CM = 30;
/** how fast the feeling fades once the petting stops (level per second) */
const FADE = 0.25;
/** happy from this level up; content again (eyes open) below RELAX: a few seconds after it stops */
const HAPPY = 1, RELAX = 0.35, MAX = 1.3;
/** a heart every this many seconds while happy, each living LIFE seconds */
const EVERY = 0.55, LIFE = 1.9;

interface Heart {
  born: number;
  /** spawn point, CSS px, and the size the heart grows to */
  x: number;
  y: number;
  size: number;
  sway: number;
}

export class Petting {
  /** 0 at rest; 1 and up is happy */
  level = 0;
  happy = false;
  /** eyes closing with pleasure, 0..1 (eased) */
  content = 0;
  readonly hearts: Heart[] = [];
  private clock = 0;
  private nextHeart = 0;
  private seed = 1;

  /** the pointer moved `cm` (head cm) over the crown */
  stroke(cm: number): void {
    if (cm > 0) this.level = Math.min(MAX, this.level + cm / HAPPY_CM);
  }

  /** one frame; `anchor` is where the hearts start (CSS px, beside the crown), `cmPx` the head's scale */
  update(dt: number, anchor: [number, number] | null, cmPx: number): void {
    this.clock += dt;
    this.level = Math.max(0, this.level - FADE * dt);
    if (this.level >= HAPPY) this.happy = true;
    else if (this.level < RELAX) this.happy = false;
    this.content += ((this.happy ? 1 : 0) - this.content) * Math.min(1, dt * (this.happy ? 5 : 3));
    if (this.happy && anchor && this.clock >= this.nextHeart) {
      this.nextHeart = this.clock + EVERY;
      const r = this.rand();
      this.hearts.push({ born: this.clock, x: anchor[0] + (r - 0.5) * 1.6 * cmPx, y: anchor[1], size: Math.min(44, Math.max(14, (1.9 + 0.6 * this.rand()) * cmPx)), sway: this.rand() * Math.PI * 2 });
    }
    while (this.hearts.length && this.clock - this.hearts[0].born > LIFE) this.hearts.shift();
  }

  /** the hearts, rising and swaying and fading; `side` is the way they drift (towards the face) */
  draw(ctx: CanvasRenderingContext2D, side: 1 | -1, cmPx: number): void {
    for (const h of this.hearts) {
      const u = (this.clock - h.born) / LIFE;
      const pop = Math.min(1, u / 0.12);
      const s = h.size * (0.35 + 0.65 * (1 - (1 - pop) ** 3)) * (1 + 0.15 * u);
      const x = h.x + side * u * 1.5 * cmPx + Math.sin(h.sway + u * 7) * 0.35 * cmPx;
      const y = h.y - u * 4 * cmPx;
      heart(ctx, x, y, s, V.pink, Math.min(1, (1 - u) / 0.4));
    }
  }

  private rand(): number {
    // small LCG: the hearts' scatter needs no more
    this.seed = (this.seed * 16807) % 2147483647;
    return this.seed / 2147483647;
  }
}

/** a heart outline `size` px tall centred on (x, y) */
export function heart(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, color: string, alpha: number): void {
  if (alpha <= 0 || size <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1.2, size / 16);
  ctx.lineJoin = "round";
  ctx.beginPath();
  const k = size / 32;
  for (let i = 0; i <= 48; i++) {
    const t = (i / 48) * Math.PI * 2;
    const hx = 16 * Math.sin(t) ** 3;
    const hy = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    const px = x + hx * k, py = y - (hy + 2.5) * k;
    if (i) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
  }
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}
