import { useEffect, useRef } from "react";

export interface CanvasFrame {
  ctx: CanvasRenderingContext2D;
  w: number; // css px
  h: number;
  dpr: number;
  dt: number;
  now: number;
}

/** Canvas with devicePixelRatio scaling, resize observer and a rAF draw loop (ctx pre-scaled to css px). */
export function useCanvasLoop(draw: (f: CanvasFrame) => void) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    let w = 0, h = 0, dpr = 1;
    const ro = new ResizeObserver(() => {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = r.width;
      h = r.height;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
    });
    ro.observe(canvas);
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (w > 0 && h > 0) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawRef.current({ ctx, w, h, dpr, dt, now });
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);
  return ref;
}

/** Canvas font in the Hershey stack (heavier weights map to its medium cut; IPA falls back to Archivo). */
export function font(size: number, weight = 450, width = 100): string {
  void width;
  return `${weight >= 550 ? 500 : 300} ${size}px "Hershey", "Archivo", system-ui, sans-serif`;
}
