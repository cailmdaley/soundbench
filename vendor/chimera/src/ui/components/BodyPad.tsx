import { useRef } from "react";
import { useStore } from "../store.ts";
import { bodyLabel, PRESETS, sameBody, type BodyMix } from "../body.ts";

const W = 64, H = 32, P = 4;
const STEP = 0.05;
const q = (x: number) => Math.min(1, Math.max(0, Math.round(x / STEP) * STEP));

/** Pad position (0..1 across, 0..1 down) of a body: male top left, female top right, child below. */
function toUV(b: BodyMix): [number, number] {
  return [0.5 + (b.female - 0.5) * (1 - b.child), b.child];
}

/** The body at a pad position (clamped into the triangle; at the child corner `female` is kept). */
function fromUV(u: number, v: number, prev: BodyMix): BodyMix {
  const child = q(v);
  const female = child < 1 ? q(0.5 + (u - 0.5) / (1 - Math.min(v, 0.999))) : prev.female;
  return { female, child };
}

const X = (u: number) => P + u * (W - 2 * P);
const Y = (v: number) => P + v * (H - 2 * P);

/**
 * The body as a point in a triangle of the three calibrated bodies (a grid of 5 % steps). Dragging
 * shows the body at once and rebuilds the voice on release (a new airway, as a preset switch
 * does); the arrow keys move it a step at a time (left/right toward male/female, down toward child).
 */
export function BodyPad() {
  const body = useStore((s) => s.body);
  const previewBody = useStore((s) => s.previewBody);
  const setBody = useStore((s) => s.setBody);
  const drag = useRef<{ id: number; at: BodyMix } | null>(null);

  const at = (e: React.PointerEvent): BodyMix => {
    const r = (e.currentTarget as Element).getBoundingClientRect();
    const u = ((e.clientX - r.left) / r.width) * (W / (W - 2 * P)) - P / (W - 2 * P);
    const v = ((e.clientY - r.top) / r.height) * (H / (H - 2 * P)) - P / (H - 2 * P);
    return fromUV(Math.min(1, Math.max(0, u)), Math.min(1, Math.max(0, v)), drag.current?.at ?? useStore.getState().body);
  };
  const move = (b: BodyMix) => {
    if (!drag.current || sameBody(b, drag.current.at)) return;
    drag.current.at = b;
    previewBody(b);
  };
  const end = () => {
    const d = drag.current;
    drag.current = null;
    if (d) setBody(d.at);
  };

  const [u, v] = toUV(body);
  const grid: string[] = [];
  for (const k of [0.25, 0.5, 0.75]) {
    grid.push(`M${X(0.5 - 0.5 * (1 - k))} ${Y(k)}H${X(0.5 + 0.5 * (1 - k))}`); // equal age
    grid.push(`M${X(k)} ${Y(0)}L${X(0.5)} ${Y(1)}`); // equal male/female share
  }
  return (
    <div
      className="body-pad"
      role="slider"
      tabIndex={0}
      aria-label="Body between male, female and child"
      aria-valuetext={bodyLabel(body)}
      aria-valuenow={Math.round(body.child * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      title={`${bodyLabel(body)} — drag between male, female and child; arrow keys step`}
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        (e.currentTarget as Element).setPointerCapture(e.pointerId);
        drag.current = { id: e.pointerId, at: useStore.getState().body };
        move(at(e));
      }}
      onPointerMove={(e) => drag.current?.id === e.pointerId && move(at(e))}
      onPointerUp={end}
      onPointerCancel={end}
      onLostPointerCapture={end}
      onKeyDown={(e) => {
        const b = useStore.getState().body;
        const next =
          e.key === "ArrowLeft" ? { ...b, female: q(b.female - STEP) }
          : e.key === "ArrowRight" ? { ...b, female: q(b.female + STEP) }
          : e.key === "ArrowUp" ? { ...b, child: q(b.child - STEP) }
          : e.key === "ArrowDown" ? { ...b, child: q(b.child + STEP) }
          : e.key === "Home" ? PRESETS.male
          : null;
        if (!next) return;
        e.preventDefault();
        if (!sameBody(next, b)) setBody(next);
      }}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} aria-hidden="true">
        <path d={grid.join("")} className="body-grid" />
        <path d={`M${X(0)} ${Y(0)}H${X(1)}L${X(0.5)} ${Y(1)}Z`} className="body-frame" />
        <circle cx={X(u)} cy={Y(v)} r={3} className="body-dot" />
      </svg>
    </div>
  );
}
