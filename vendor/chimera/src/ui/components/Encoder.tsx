import { useId, useRef, useState, type ReactNode } from "react";
import { SLOTS } from "../colors.ts";

export interface EncoderSpec {
  label: string;
  value: number;
  min: number;
  max: number;
  /** logarithmic mapping (pitch, glide) */
  log?: boolean;
  onChange: (v: number) => void;
  /** big numeral and its unit */
  format: (v: number) => { value: string; unit: string };
}

const TICKS = 10;
const A0 = (225 * Math.PI) / 180;
const SPAN = (270 * Math.PI) / 180;

/**
 * OP-1 style encoder: a dial and a big numeral in its slot colour, over a native range input
 * that does the work (label, keyboard incl. Home/End and Page keys, aria-valuetext). Dragging the
 * dial is relative: up or right raises the value. On touch, vertical swipes scroll the page
 * (touch-action: pan-y) and only horizontal drags edit.
 */
export function Encoder(props: EncoderSpec & { slot: number }) {
  const { label, value, min, max, log, onChange, format, slot } = props;
  const id = useId();
  const color = SLOTS[slot % 4];
  const toPos = (v: number) => (log ? (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min)) : (v - min) / (max - min));
  const fromPos = (p: number) => {
    const u = Math.min(1, Math.max(0, p));
    return log ? Math.exp(Math.log(min) + u * (Math.log(max) - Math.log(min))) : min + u * (max - min);
  };
  const pos = Math.min(1, Math.max(0, toPos(value)));
  const shown = format(value);
  const drag = useRef<{ x: number; y: number; pos: number; id: number; touch: boolean } | null>(null);
  const at = (u: number, r: number) => [Math.cos(A0 - u * SPAN) * r, -Math.sin(A0 - u * SPAN) * r];
  const [x0, y0] = at(0, 30), [x1, y1] = at(pos, 30), [px, py] = at(pos, 23);
  const arc = pos > 0.002 ? `M${x0.toFixed(2)} ${y0.toFixed(2)}A30 30 0 ${pos * 270 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}` : "";
  return (
    <div className="encoder" style={{ "--slot": color } as React.CSSProperties}>
      <label className="enc-label" htmlFor={id}>
        {label}
      </label>
      <div
        className="enc-dial"
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          (e.currentTarget as Element).setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, y: e.clientY, pos, id: e.pointerId, touch: e.pointerType === "touch" };
          document.getElementById(id)?.focus({ preventScroll: true });
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d || d.id !== e.pointerId) return;
          const dx = e.clientX - d.x, dy = e.clientY - d.y;
          // a vertical touch swipe is the page scrolling (pan-y), not an edit: touch uses dx only,
          // past a small dead zone; mouse and pen also take vertical movement
          if (d.touch && Math.abs(dx) < 6) return;
          const delta = (d.touch ? dx : dx - dy) / (e.shiftKey ? 600 : 160);
          onChange(fromPos(d.pos + delta));
        }}
        onPointerUp={() => (drag.current = null)}
        onPointerCancel={() => (drag.current = null)}
        onLostPointerCapture={() => (drag.current = null)}
      >
        <svg viewBox="-42 -42 84 84" aria-hidden="true">
          {Array.from({ length: TICKS + 1 }, (_, i) => {
            const [ax, ay] = at(i / TICKS, 35), [bx, by] = at(i / TICKS, 39);
            return <line key={i} x1={ax} y1={ay} x2={bx} y2={by} className="enc-tick" />;
          })}
          <circle r="30" className="enc-ring" />
          {arc && <path d={arc} className="enc-arc" />}
          <line x1="0" y1="0" x2={px} y2={py} className="enc-pointer" />
          <circle r="2.5" className="enc-hub" />
        </svg>
        <input
          id={id}
          className="enc-input"
          type="range"
          min={0}
          max={1000}
          step={1}
          value={Math.round(pos * 1000)}
          onChange={(e) => onChange(fromPos(Number(e.target.value) / 1000))}
          aria-valuetext={`${shown.value} ${shown.unit}`.trim()}
        />
      </div>
      <output className="enc-value" htmlFor={id}>
        {shown.value}
      </output>
      <span className="enc-unit">{shown.unit}</span>
    </div>
  );
}

export interface EncoderPage {
  name: string;
  encoders: EncoderSpec[];
}

/**
 * Four encoders at a time, with page tabs. The page shown is presentation only. Slots keep their
 * positions across pages (so a colour stays put); a single short page can use fewer columns.
 */
export function EncoderPages(props: { pages: EncoderPage[]; footer?: ReactNode; label: string; columns?: number }) {
  const cols = props.columns ?? 4;
  const [page, setPage] = useState(0);
  const p = props.pages[Math.min(page, props.pages.length - 1)];
  return (
    <div className="enc-pages">
      {props.pages.length > 1 && (
        <div className="enc-tabs" role="group" aria-label={`${props.label} pages`}>
          {props.pages.map((pg, i) => (
            <button key={pg.name} type="button" className={`enc-tab${i === page ? " on" : ""}`} aria-pressed={i === page} onClick={() => setPage(i)}>
              {pg.name}
            </button>
          ))}
        </div>
      )}
      <div className={`enc-row cols-${cols}`} role="group" aria-label={p.name}>
        {Array.from({ length: cols }, (_, i) => i).map((i) => (p.encoders[i] ? <Encoder key={`${p.name}-${p.encoders[i].label}`} {...p.encoders[i]} slot={i} /> : <div key={`empty-${i}`} className="encoder empty" aria-hidden="true" />))}
      </div>
      {props.footer && <div className="enc-footer">{props.footer}</div>}
    </div>
  );
}
