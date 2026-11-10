# Mandolin playing contract

Four courses, numbered high to low: **1 E5 (76), 2 A4 (69), 3 D4 (62), 4 G3 (55)**.
Each course excites two independent coupled strings with 1.5–3 cents detuning and slightly different decay rates.
The pick crosses them **2.4 ms apart**, with **54%/46% excitation**, different comb/pluck positions and independently seeded position jitter.
Downstrokes pick primary then secondary; upstrokes reverse that order.
Playable range is open pitch through fret 20: MIDI **55–96**, depending on course.
All pitches below are sounding MIDI pitches; don't transpose the mandolin staff.

```toml
[instruments]
mandolin = { model = "mandolin", gain_db = 0, pan = 0 }
[mandolin_strings]
lead = 1 # optional selector for a separate MIDI Voice, not a channel
[mandolin_model]
bend_range = 12 # pitchwheel semitones, fixed; MIDI RPN isn't interpreted
```

Selectors use full MIDI track names, colon components or one-based note-track ordinals.
No track-name gesture conventions or keyswitch notes exist.
Without a pin, notes choose the lowest-fret free course, then replace the oldest occupied course.
Chords need distinct playable courses; separate pinned Voices make fingering unambiguous.
**Every note needs a note-off**; it damps open courses too.
CC64 doesn't sustain this model.

## MIDI gestures

CCs are per channel **inside each note-bearing MIDI track**.
Their values are latched at note-on; reset the gesture after its note or it carries forward.
Put the CC before the note-on at the same tick.
LilyPond's normal notation supplies pitches, note lengths and velocities, but doesn't infer tremolo/chop/legato from printed marks.
A generator or MIDI postprocessor must insert these messages.

| Gesture | Exact request |
|---|---|
| Single note / double stop | CC23=0, CC24=0; ordinary note-on/off. CC25=0–63 selects downstroke, CC25=64–127 selects upstroke (default down). Double stops are simultaneous notes on two courses. |
| Tremolo | CC23=64; one held note. CC20 controls attacks/s: `4 + 20*value/127` (default 51 → 12.03/s). CC21 is evenness `value/127` (default 114). CC22 is alternate downstroke accent `1 + 0.5*value/127` (default 30 → 1.118). |
| Chop | CC23=127, CC24=0; chord note-ons receive an explicit 20 ms left-hand choke. Write these on beats 2 and 4. Note-offs still required. Reset CC23=0 afterward. |
| Crosspicking | CC23=0; write each eighth/sixteenth note explicitly on alternating courses. No pattern is generated. |
| Hammer-on | Pin the Voice/course, hold the source until the destination onset, then CC24=1 and a higher destination note. The destination takes the course without a full pick attack. |
| Pull-off | Same, CC24=2 and a lower destination note. Reset CC24=0 before the next picked note. |
| Slide | Ordinary held note plus pitchwheel control points. Zero=score pitch; −8192≈−12 semitones, +8191≈+12 by default. Linear ramps connect successive messages; the note-on pitch holds the last preceding wheel value. Keep every target within the selected course's fret range. |

Tremolo and legato are separate gestures; don't combine CC23=64/127 with nonzero CC24.
Pitchwheel controls every sounding note on that track/channel, including a double stop.
Give independent pitch lines separate tracks/channels.
For a delayed slide, emit a zero wheel point at the slide start before the destination point.
Without it, the ramp starts at note-on.

## Generator API

Import `soundbench.scripts.models.folk_strings`.
Times are seconds; course numbers are 1–4; tokens must be unique within a performance.
`note(on, off, pitch, velocity, course, token, owner=0, kind=0, direction='down')` emits attack and release.
Use `direction='up'` to reverse crossing order; tremolo alternates it automatically.
For raw 11-column events, attack column 7 is 0 down / 1 up; on pitch ramps it retains the ease-shape meaning.
Kinds 3 and 4 are hammer and pull destinations.
`bar(on, [(0, start_pitch), (duration, target_pitch)], course, token)` adds continuous absolute-pitch ramps to that token.
`tremolo(on, off, pitch, velocity, course, token, rate=12, evenness=.9, accent=1.12, seed=817)` emits alternate strokes and a final release; reserve at least `ceil(duration*rate*1.2)` tokens after its base token.
`chop(on, pitches, courses, config, velocity=95, token=0)` emits a 12 ms downstroke and 20 ms per-course choke; reserve four tokens.
`render(events, seconds, owners=1, config=load_config('mandolin'), instrument='mandolin')` returns owner-by-sample mono stems at 48 kHz.
The renderer adds deliberate final damping in its tail; generator callers can use `models.guitar_release.finish(events, at)`.

## Musical ranges

Velocity **65–100** works for lead, **65–90** for chops, **55–85** for tremolo.
Use tremolo around **10–15 attacks/s**, evenness **0.8–0.95**, accent **1.05–1.2**.
The picking rate has seeded slow drift around the requested rate, plus per-stroke timing scatter.
At evenness 0.9, the slow drift is bounded by ±2.4%; downstroke velocities receive the requested accent.
Evenness 1 is the explicit machine-even option: it disables drift and timing/dynamic scatter, but retains the accent.
Low velocities deliberately soften the high-frequency attack as well as reduce level.
A high lead on courses 1–2 and two-course chops on 3–4 can coexist.
A full four-course chop replaces a sustained lead on the same instrument: use a second independently rendered mandolin for a full accompanying chop.
Fast crosspicking needs overlapping note tails rather than tiny separated MIDI gates.
Extreme tremolo rates, identical full-velocity repetitions and large fast slides expose the model's regularity.
