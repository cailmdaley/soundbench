# Dobro playing contract

Square-neck lap-style preset in **GBDGBD** (low to high).
String numbers run high to low: **1 D4 (62), 2 B3 (59), 3 G3 (55), 4 D3 (50), 5 B2 (47), 6 G2 (43)**.
Playable bar range is **0–24 semitones above each open string**, continuously, without fret quantization.
The sounding MIDI range is 43–86.
The body uses a metallic resonator-mode bank; the reference fit doesn't establish square-neck construction.

```toml
[instruments]
dobro = { model = "dobro", gain_db = 0, pan = 0 }
[dobro_strings]
melody = 1 # optional: pin a separate MIDI Voice to physical string 1
[dobro_model]
bend_range = 12 # wheel range in semitones; allowed >0 through 24
vibrato = { depth_cents = 18, rate = 5.5, onset_delay = 0.25 }
# vibrato = false disables automatic bar rocking.
```

Selectors use full MIDI track names, colon components or one-based note-track ordinals.
No keyswitches, CC bar-position convention, track-name gesture suffixes or RPN bend-range changes exist.
Unpinned notes prefer the lowest-fret free string; overlapping notes can replace the oldest occupied string.
Use pinned Voices for a reproducible bar melody and rolls.
**Every note must have a note-off**; it damps even open strings.
CC64 is ignored.
For ringing rolls, lengthen the scored notes to overlap rather than using pedal.

## Continuous bar control

Use **MIDI pitchwheel** in the note's MIDI track and channel.
Its pitch is an offset from each sounding note's scored pitch: `bend_range * wheel/8192` semitones.
Zero is the scored pitch; the default extrema are −12 and just under +12 semitones.
An existing wheel value is held at note-on.
Linear continuous ramps connect subsequent messages; a point at the slide start is necessary to delay a slide beyond note-on.
Every target must remain above its string's open pitch and at or below fret 24.
A negative wheel works when the scored note is sufficiently above the open string.

Example: MIDI 62 on pinned string 1, wheel 0 at note-on, wheel 3413 at 0.7 s → **G4, about +5 semitones**, without a new pick.
Wheel 0 at 0.3 s before the destination instead keeps D4 steady for 0.3 s, then slides for 0.4 s.
**Automatic bar vibrato is on for fretted/bar-sounded notes longer than 2/3 s** (one quarter note at 90 bpm).
Open-string notes and shorter roll notes remain still.
It defaults to **±18 cents at 5.5 Hz**, beginning **250 ms after the last substantial slide arrives**, with depth growing smoothly over **400 ms**.
`[dobro_model] vibrato` accepts `depth_cents` (0–50), `rate` (1–10 Hz), `onset_delay` (0–5 s), `rise_seconds` (0.05–2 s), and `minimum_seconds` (0–10 s).
Zero depth or `vibrato = false` turns it off.
The automatic offset is added to the wheel's center-pitch curve, stays within bar range and fades over the last 80 ms before release.
For hand-written vibrato, disable the automatic articulation and write points at **25–50 Hz**, typically **±0.1–0.25 semitones at 5–6 cycles/s**.
Reset the wheel before a new scored phrase unless its starting pitch intentionally includes the held bend.
Pitchwheel messages after note-off do not move that note.

One channel moves all its sounding notes in parallel: this is a straight bar.
For a slant, put strings in separate pinned MIDI tracks/channels and send different wheel curves.
The model permits independent targets; it doesn't enforce physically reachable bar angles.
The friction signal follows explicit bar travel, is seeded and vanishes while the bar is still.
Slides of at least half a semitone lasting at least 40 ms also receive a light colored, resonant squeak.
Automatic bar rocking doesn't add a hiss or squeak to every vibrato cycle.
Its gain is an unfitted prior: there is no identified steel-bar-noise recording in the reference set.

## Picks and rolls

Ordinary note-on/off means finger/thumb-pick excitation.
Write rolls as explicit note sequences, e.g. strings **6–4–2–5–3–1** or **6–4–3**, with overlapping releases.
Useful melody velocities are **70–100**, bass/thumb **60–85**, middle/index **45–70**.
The deployed preset has one common pick response; it doesn't infer a thumb from a low MIDI pitch.
CC24=1/2 can request the guitar engine's hammer/pull mechanics, but these aren't idiomatic bar articulations: use pitchwheel instead.
CC20–23 tremolo/chop controls aren't used for dobro; CC23 must remain zero.
Natural/artificial harmonics aren't implemented.

## Generator API

Import `soundbench.scripts.models.folk_strings`.
`note(on, off, pitch, velocity, string, token, owner=0)` emits attack/release in seconds.
`bar(on, [(0, start_pitch), (.7, target_pitch), ...], string, token, owner=0)` appends **absolute fractional MIDI-pitch** ramps for the same token.
Its first pitch must equal the plucked starting pitch; times must increase strictly.
A constant-pitch segment holds the bar still.
Write one curve per string for a slant.
`vibrato(on, off, pitch, string, token, depth_cents=18, rate=5.5, onset_delay=.25, rise_seconds=.4)` supplies an explicit growing vibrato articulation for a stationary bar note.
Use `render(..., vibrato=False)` when supplying that explicit curve to avoid applying it twice.
`render(events, seconds, owners=1, config=load_config('dobro'), instrument='dobro', vibrato=None)` produces 48 kHz owner-by-sample mono stems, adding default bar vibrato.
Pass the same control dictionary as the MIDI model options to override it.
Every independent note token must be unique.
Direct callers can use `models.guitar_release.finish(events, at)` for deliberate end damping.

Large instantaneous wheel changes, vibrato beyond half a semitone, and full-velocity rolls are conspicuously synthetic.
A straight bar needs matching fret offsets on accompanying strings; the scheduler doesn't silently correct an impossible score.
G/C/D bar positions 12/5/7, slow arriving slides, small vibrato and moderate roll velocities sit naturally.
