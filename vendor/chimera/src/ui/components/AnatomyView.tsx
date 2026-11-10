import { useEffect, useRef, useState } from "react";
import { AnatomyRenderer, type FaceView, type HandleId, type Slot } from "../render/anatomy.ts";
import { Glow } from "../render/vector.ts";
import { engine } from "../../audio/AudioEngine.ts";
import { sameBody } from "../body.ts";
import { useStore } from "../store.ts";
import { VoiceModel } from "../analysis.ts";
import { useModel } from "../useModel.ts";
import { V } from "../colors.ts";
import type { HumanArtic, HumanFrame } from "../../wasm/types.ts";
import type { TelemetryMessage } from "../../worklet/protocol.ts";

/** What the four readouts under the chin show; a mode can pass its own. */
export type SlotsFn = (m: VoiceModel, a: HumanArtic, f: HumanFrame, formants: number[], tele: TelemetryMessage | null) => Slot[];
/** The small lines under the formants; a mode can pass its own. */
export type InfoFn = (m: VoiceModel, f: HumanFrame, tele: TelemetryMessage | null) => string[];

/** drawn over the face each frame, before the glow (Sing and Speak write the words at the mouth) */
export type OverlayFn = (ctx: CanvasRenderingContext2D, r: FaceView, m: VoiceModel, f: HumanFrame, a: HumanArtic, size: { w: number; h: number; dt: number }) => void;

/** the tract's length and mouth opening, then the live pitch and lungs */
export const tractInfo: InfoFn = (_m, f, tele) => {
  const info = [`Tract ${f.length.toFixed(1)} cm`, `Mouth ${f.lipArea.toFixed(1)} cm²`];
  if (tele) info.push(`F0 ${tele.t.f0.toFixed(1)} Hz`, `Lungs ${tele.t.pLung.toFixed(1)} cmH₂O`);
  return info;
};

/** the calibrated vowel nearest to (F1, F2), same metric as the vowel map */
export function nearestVowel(m: VoiceModel, F1: number, F2: number): string {
  let best = "", bd = Infinity;
  for (const v of m.vowels) {
    const d = Math.hypot(Math.log(F1 / v.F1) * 2.2, Math.log(F2 / v.F2));
    if (d < bd) {
      bd = d;
      best = v.ipa;
    }
  }
  return best;
}

/** Play mode: the tongue as its nearest vowel, then jaw, lips and soft palate */
export const articulatorSlots: SlotsFn = (m, a, f, formants) => [
  formants.length >= 2 ? { label: "Tongue", value: nearestVowel(m, formants[0], formants[1]), color: V.green, ipa: true } : { label: "Tongue", value: "-", color: V.green },
  { label: "Jaw", value: a.jaw.toFixed(1), unit: "cm", color: V.red },
  { label: "Lips", value: a.lipAp.toFixed(1), unit: "cm", color: V.pink },
  f.velumPort > 0.02 ? { label: "Soft palate", value: f.velumPort.toFixed(2), unit: "cm²", color: V.white } : { label: "Soft palate", value: "Shut", color: V.white },
];

export function AnatomyView(props: { slots?: SlotsFn; info?: InfoFn; showFormants?: boolean; overlay?: OverlayFn }) {
  const overlayRef = useRef<OverlayFn | undefined>(props.overlay);
  overlayRef.current = props.overlay;
  const formantsRef = useRef(props.showFormants ?? true);
  formantsRef.current = props.showFormants ?? true;
  const slotsRef = useRef<SlotsFn>(props.slots ?? articulatorSlots);
  slotsRef.current = props.slots ?? articulatorSlots;
  const infoRef = useRef<InfoFn>(props.info ?? tractInfo);
  infoRef.current = props.info ?? tractInfo;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const model = useModel();
  const [hover, setHover] = useState<HandleId | null>(null);
  const dragRef = useRef<{ id: HandleId; start: { x: number; y: number }; artic: HumanArtic } | null>(null);
  const hoverRef = useRef<HandleId | null>(null);
  hoverRef.current = hover;
  const rendererRef = useRef<AnatomyRenderer | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const wrap = wrapRef.current!;
    const r = new AnatomyRenderer();
    const glow = new Glow();
    rendererRef.current = r;
    let w = 0, h = 0, dpr = 1;
    const fit = () => {
      const b = wrap.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = b.width;
      h = b.height;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
      r.resize(w, h);
    };
    const ro = new ResizeObserver(fit);
    ro.observe(wrap);
    let raf = 0;
    let last = performance.now();
    let fTick = 0;
    let formants: number[] = [];
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      // the face's expression keeps real time (a held note lasts as long at a low frame rate)
      const elapsed = Math.min(0.5, (now - last) / 1000);
      last = now;
      if (Math.min(2, window.devicePixelRatio || 1) !== dpr) fit();
      const st = useStore.getState();
      const tele = engine.latest && sameBody(engine.latest.body, st.body) ? engine.latest : null;
      const a = tele ? VoiceModel.articFromArray(tele.t.artic) : st.artic;
      r.face.update(tele, elapsed);
      const m = model.current;
      if (m && w > 0 && h > 0) {
        const f = m.frame(a);
        if (++fTick % 4 === 0 || formants.length === 0) formants = m.transfer(a).formants;
        const info = infoRef.current(m, f, tele);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        r.draw(ctx, m, f, a, tele, st.view, { hover: hoverRef.current, drag: dragRef.current?.id ?? null, dt, formants: formantsRef.current ? formants : [], slots: slotsRef.current(m, a, f, formants, tele), info: info.map((s) => s.toUpperCase()) });
        overlayRef.current?.(ctx, r, m, f, a, { w, h, dt });
        glow.apply(canvas, ctx, dpr);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [model]);

  const pick = (e: React.PointerEvent): HandleId | null => {
    const r = rendererRef.current;
    const m = model.current;
    if (!r || !m) return null;
    const rect = canvasRef.current!.getBoundingClientRect();
    const p = r.toBody(e.clientX - rect.left, e.clientY - rect.top);
    const st = useStore.getState();
    const tele = engine.latest && sameBody(engine.latest.body, st.body) ? engine.latest : null;
    const a = tele ? VoiceModel.articFromArray(tele.t.artic) : st.artic;
    const f = m.frame(a);
    let best: HandleId | null = null;
    let bd = 22 / r.pxPerCm; // 22 px hit radius
    for (const h of r.handles(m, f, a)) {
      const d = Math.hypot(h.pos.x - p.x, h.pos.y - p.y);
      if (d < bd) {
        bd = d;
        best = h.id;
      }
    }
    return best;
  };

  const onDown = (e: React.PointerEvent) => {
    const id = pick(e);
    if (!id) return;
    const r = rendererRef.current!;
    const rect = canvasRef.current!.getBoundingClientRect();
    const p = r.toBody(e.clientX - rect.left, e.clientY - rect.top);
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = { id, start: p, artic: { ...useStore.getState().artic } };
    e.preventDefault();
  };

  const onMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    const r = rendererRef.current;
    const m = model.current;
    if (!d) {
      const id = pick(e);
      if (id !== hover) setHover(id);
      return;
    }
    if (!r || !m) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const p = r.toBody(e.clientX - rect.left, e.clientY - rect.top);
    const t = m.tract;
    const k = t.scale;
    const dx = (p.x - d.start.x) / k;
    const dy = (p.y - d.start.y) / k;
    const set = useStore.getState().setArtic;
    const a0 = d.artic;
    switch (d.id) {
      case "tongue": {
        const q = t.unP(p.x, p.y);
        set({ tbX: q.x, tbY: q.y });
        break;
      }
      case "tip": {
        const q = t.unP(p.x, p.y);
        set({ ttX: q.x, ttY: q.y }, false);
        break;
      }
      case "jaw":
        set({ jaw: clamp(a0.jaw - dy, -0.1, 2.6) });
        break;
      case "lips":
        if (e.shiftKey) set({ lipWidth: clamp(a0.lipWidth + dx * 0.6, 0, 1) });
        else set({ lipAp: clamp(a0.lipAp - dy * 1.2, -0.4, 2.4), lipProt: clamp(a0.lipProt + dx, -0.2, 1.5) });
        break;
      case "velum":
        set({ velum: clamp(a0.velum - dy * 0.9, 0, 1.4) });
        break;
      case "larynx":
        set({ larynx: clamp(a0.larynx + dy, -1.6, 1.6) });
        break;
    }
  };

  const onUp = (e: React.PointerEvent) => {
    if (dragRef.current) (e.target as Element).releasePointerCapture?.(e.pointerId);
    dragRef.current = null;
  };

  const onDouble = (e: React.MouseEvent) => {
    const id = pick(e as unknown as React.PointerEvent);
    if (id === "tip") useStore.getState().setArtic({}, true);
  };

  return (
    <div ref={wrapRef} className="anatomy" data-hover={hover ?? undefined}>
      <canvas
        ref={canvasRef}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onDoubleClick={onDouble}
        aria-label="Vocal tract cross-section. Drag the tongue, tongue tip, jaw, lips, soft palate or larynx handles."
        role="img"
      />
    </div>
  );
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
