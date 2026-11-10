import { useEffect, useRef } from "react";
import { Glow } from "./render/vector.ts";
import type { CanvasFrame } from "./useCanvas.ts";

/**
 * Canvas for a vector-display screen: devicePixelRatio scaling (capped at 2), resize observer,
 * a rAF loop that clears to black, calls draw() in CSS pixels, then adds the glow.
 * Screens draw every frame: telemetry, afterglow and scrolling views change outside the store.
 */
export function usePhosphor(draw: (f: CanvasFrame) => void, opts: { sigma?: number; alpha?: number } = {}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const { sigma, alpha } = opts;
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const glow = new Glow(sigma, alpha);
    let w = 0, h = 0, dpr = 1;
    const fit = () => {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = r.width;
      h = r.height;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
    };
    const ro = new ResizeObserver(fit);
    ro.observe(canvas);
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      // a window moved to a display with another scale changes DPR without a resize
      if (Math.min(2, window.devicePixelRatio || 1) !== dpr) fit();
      if (w > 0 && h > 0) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawRef.current({ ctx, w, h, dpr, dt, now });
        glow.apply(canvas, ctx, dpr);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [sigma, alpha]);
  return ref;
}
