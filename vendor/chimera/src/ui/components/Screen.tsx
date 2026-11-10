import type { ButtonHTMLAttributes, ReactNode } from "react";

/**
 * One screen on the wall: a black panel with a thin frame and a single-stroke title.
 * The body always has a positive size (see .screen-body in styles.css); canvases fill it.
 */
export function Screen(props: { title: string; sub?: string; focus?: boolean; className?: string; tools?: ReactNode; children: ReactNode; label?: string }) {
  const { title, sub, focus, className, tools, children, label } = props;
  return (
    <section className={`screen${focus ? " focus" : ""}${className ? " " + className : ""}`} aria-label={label ?? title}>
      <header className="screen-head">
        <h2 className="screen-title">{title}</h2>
        {sub && <span className="screen-sub">{sub}</span>}
        {tools && <div className="screen-tools">{tools}</div>}
      </header>
      <div className="screen-body">{children}</div>
    </section>
  );
}

/** A framed caps button; `on` lights it and sets aria-pressed (a toggle) unless it's a plain action. */
export function Chip(props: ButtonHTMLAttributes<HTMLButtonElement> & { on?: boolean; toggle?: boolean; tone?: "slot0" | "slot1" | "slot2" | "slot3" | "amber" }) {
  const { on, toggle = true, tone, className, children, ...rest } = props;
  return (
    <button type="button" {...rest} className={`chip${on ? " on" : ""}${tone ? " " + tone : ""}${className ? " " + className : ""}`} aria-pressed={toggle ? !!on : undefined}>
      {children}
    </button>
  );
}
