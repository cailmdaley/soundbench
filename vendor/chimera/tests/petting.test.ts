/**
 * Petting a head: a short wiggle over the crown makes it happy (eyes closing, hearts), a single
 * pass doesn't, and it settles a few seconds after the petting stops.
 */
import { describe, expect, test } from "bun:test";
import { Petting } from "../src/ui/render/petting.ts";

/** wiggle at `cmPerS` (head cm per second) for `s` seconds, at 60 frames a second */
function wiggle(p: Petting, cmPerS: number, s: number) {
  for (let t = 0; t < s; t += 1 / 60) {
    p.stroke(cmPerS / 60);
    p.update(1 / 60, [100, 100], 20);
  }
}
const rest = (p: Petting, s: number) => {
  for (let t = 0; t < s; t += 1 / 60) p.update(1 / 60, [100, 100], 20);
};

describe("Petting", () => {
  test("a second and a half of ordinary wiggling makes it happy: eyes close, hearts float up", () => {
    const p = new Petting();
    wiggle(p, 30, 1.6);
    expect(p.happy).toBe(true);
    rest(p, 0.6);
    expect(p.content).toBeGreaterThan(0.9);
    expect(p.hearts.length).toBeGreaterThan(0);
  });

  test("passing over the head once does nothing", () => {
    const p = new Petting();
    wiggle(p, 30, 0.4); // 12 cm: across the crown and off
    rest(p, 1);
    expect(p.happy).toBe(false);
    expect(p.content).toBeLessThan(0.01);
    expect(p.hearts.length).toBe(0);
  });

  test("it settles a few seconds after the petting stops, and the hearts finish", () => {
    const p = new Petting();
    wiggle(p, 40, 4);
    expect(p.happy).toBe(true);
    rest(p, 2);
    expect(p.happy).toBe(true); // it lingers
    rest(p, 4);
    expect(p.happy).toBe(false);
    rest(p, 2);
    expect(p.content).toBeLessThan(0.01);
    expect(p.hearts.length).toBe(0);
  });
});
