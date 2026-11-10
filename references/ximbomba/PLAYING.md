# Ximbomba: score contract

```toml
[instruments]
ximbomba = { model = "ximbomba", gain_db = 3, pan = 0 }

```

The default preset is fitted to a recording of a Mallorcan ximbomba playing alone.
`preset = "austria"` under `[ximbomba_model]` selects a fit to an Austrian friction drum, whose hand lifts off between strokes.
Use one named pitched MIDI Voice, not a GM drum map.
No keyswitches, pitch bends, CC64 sustain or track-name articulation suffixes are used.

| MIDI | LilyPond absolute pitch | Gesture |
| --- | --- | --- |
| 48 (C3) | `c` | Upward hand stroke |
| 49 (C♯3) | `cis` | Downward hand stroke |

**Note length is hand-stroke duration.**
Strokes less than 50 ms apart are one run of the hand on the cane: the head rings on through each turn, so legato strokes join into one continuous groan, the way Mallorcan players rub.
A note-off followed by a longer rest lifts the hand and lets the head release.
Each stroke is a pitched groan: the wet hand sticks to the cane and slips once per cycle of the head, so the stroke sounds a harmonic tone.
The pot is open through its base, so its fundamental comes out weak and the third and fourth harmonics carry the groan, as in the Mallorcan field recordings measured.
At CC2 89 the fitted preset groans at a steady 75–76 Hz, the recorded drum's pitch, with harmonics 2–5 strongest and little above 400 Hz; the prior (80 Hz head, closed pot) groans at about 73 Hz with a strong fundamental.
The two directions use different hand speed, profile skew and grip; they aren't polarity copies.
Velocity 1–127 sets radiated intensity continuously; it does not change hand speed or the friction regime.
Velocity 55–95 and durations 0.25–0.75 s are useful for beat strokes.
The MIDI onset means **turn the hand**: with the fitted preset the hand reverses in 91 ms (the prior: 60 ms), scraping as it sticks and breaks away, then eases from full speed to 43% of it (the prior: 70%) by the next turn while keeping its grip.
With the `austria` preset the hand instead digs in (speed peaks 15% of the way through, `stroke_peak`), fades toward note-off and lets go, so its strokes stay separate.
All notes require explicit releases; do not tie across rests.
A stroke that starts before the previous one ends takes the hand over at its onset.

**CC2 = pressure**, 0–127, channel-local, default 0.7 (about CC2=89).
Set it before the note or move it during the note; changes apply to all currently active strokes on that channel.
0 supplies no friction and no sound.
Use **CC2 75–110** for ordinary strokes and velocity for dynamics; 89 is the default.
More pressure holds each stick longer: the groan drops in pitch, grows rougher and can break toward a subharmonic growl. It is not a volume control.

In a generator, render rows as `{"time": seconds, "off": seconds, "pitch": 48_or_49, "velocity": 1_to_127, "pressure": 0_to_1}`.
For MIDI, use `mido.Message("control_change", channel=..., control=2, value=...)`; note-on/off supply duration and intensity.
Players in Mallorcan ximbombades rub continuously: alternate C3/C♯3 with each stroke lasting until the next (the recorded players turn every 0.88 s; festival ximbombades run 140–170 strokes per minute), so the groans join into one pulsing drone.
For a slow 3/4 glosa, the same alternation on all three beats works; leave a gap of more than 50 ms where the hand should lift.
Binary 2/4 uses the same beat strokes; the engine doesn't impose a meter.

The fitted preset matches that recording's spectral envelope, sounding pitch, how steadily it holds and how continuously it carries across the turns.
