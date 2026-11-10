import { useId } from "react";

export function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  log?: boolean;
}) {
  const id = useId();
  const { label, value, min, max, step, onChange, format, log } = props;
  const toPos = (v: number) => (log ? (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min)) : (v - min) / (max - min));
  const fromPos = (p: number) => (log ? Math.exp(Math.log(min) + p * (Math.log(max) - Math.log(min))) : min + p * (max - min));
  const pos = Math.min(1, Math.max(0, toPos(value)));
  return (
    <div className="slider">
      <label htmlFor={id}>
        <span>{label}</span>
        <output>{format ? format(value) : value.toFixed(2)}</output>
      </label>
      <input
        id={id}
        type="range"
        min={0}
        max={1000}
        step={step ? Math.max(1, Math.round((step / (max - min)) * 1000)) : 1}
        value={Math.round(pos * 1000)}
        onChange={(e) => onChange(fromPos(Number(e.target.value) / 1000))}
        style={{ "--pos": `${pos * 100}%` } as React.CSSProperties}
      />
    </div>
  );
}

export function Toggle(props: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={props.checked} onChange={(e) => props.onChange(e.target.checked)} />
      <span>{props.label}</span>
    </label>
  );
}

export function Segmented<T extends string>(props: { value: T | null; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="segmented" role="radiogroup" aria-label={props.label}>
      {props.options.map((o) => (
        <button key={o.value} role="radio" aria-checked={props.value === o.value} className={props.value === o.value ? "on" : ""} onClick={() => props.onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
