# Skinpipe — composer sheet

The skinpipe is one reed, one closed-bell chanter, a membrane head and five sympathetic courses.
Its chanter and thumb share one physical body and one audio stem.
Write sounding pitch, with one MIDI track per LilyPond Voice.

```toml
[instruments]
skinpipe = { model = "skinpipe", gain_db = 0, pan = -0.1 }
# Voice skinthumb inherits the unique skinpipe binding.

[skinpipe_model]
parameters = "instruments/models/skinpipe.toml"
courses = [50, 57, 62, 66, 69]
# output_gain = 0.1
```

Use Voices named `skinpipe` and `skinthumb`, each on its own MIDI track.
An explicit skinthumb binding also works, but its gain and pan must match the chanter's.
Controls come only from the chanter track; skinthumb controls do not affect the body.
Multiple independent chanter bindings work without a thumb; a skinthumb requires exactly one chanter binding.

## Compass and tuning

The chanter accepts A3–B4, MIDI 57–71, chromatically.
It is monophonic: the latest onset changes the fingering, overlaps are allowed, and simultaneous chanter notes fail.
The default courses are D3, A3, D4, F♯4 and A4 (50, 57, 62, 66, 69).
A course pluck must match one of those exact MIDI pitches; an invalid pitch fails with its note name.
A piece can select five distinct courses in MIDI 36–84 through `skinpipe_model.courses`.

The D-relative just table is the xeremia's constructed C table rotated by two semitones, with D anchored to equal-tempered D.
It is a compositional tuning, not measured traditional tuning.
Both chanter and courses use it.
The offsets, C through B, are `[0, -11.73, 0, 0, 3.91, 0, -13.69, -1.96, 0, 1.96, 0, -15.64]` cents.

## Gestures

| Request | MIDI / score | Result |
| --- | --- | --- |
| Fingering | Chanter note, MIDI 57–71 | Continuous reed and bore history; velocity gently changes dry-path effort. |
| Repeated pitch | Scored 25–60-ms upper grace cut | The reed does not tongue; a silent scored gap holds the last fingering. |
| Bag pressure | CC2, nominal 96; bloom domain 90–108 | Changes drive; closed-head pitch sensitivity is 22 cents per relative-pressure unit. |
| Breath cut | CC24 = 0 | Bag pressure falls in 90 ms, exposing the head/course halo. |
| Restart | CC24 = 127 | Refills with the automatic short squawk. |
| Lean | CC74 = 0–127 | Smoothly raises head frequencies up to ×1.4 and reduces head T60 to ×0.25; 40-ms smoothing. |
| Damper | CC64 <64 | Lays felt on the courses: 150-ms loop T60, with a 30-ms control ramp. |
| Free courses | CC64 ≥64, default 127 | Natural course decay, nominal T60 7 / 6.2 / 5.4 / 4.7 / 4 seconds. |
| Thumb pluck | Note on skinthumb at an exact course pitch | A banjo-style pulse and pluck-position comb excite that course, with bag on or off. |
| Squawk depth | CC22, default 127 | Scales the onset's pitch rise and reed-air transient; 0 suppresses them. |
| End sag depth | CC23, default 127 | Scales the automatic 550-ms end droop after the last note-off. |
| Vibrato | CC1 | Up to ±8 cents at 5.4 Hz. |

The bag starts at the first chanter onset and naturally stops after the latest chanter note-off.
Inside that interval, ordinary rests do not stop it.
CC24 creates explicit breath cuts and restarts.
A CC24 cut exactly at the final note-off takes the 90-ms cut instead of the automatic sag.
For an unambiguous final halo, place the cut while the last note is still held, then release the note.
Leave the default nine-second render tail for the courses, or score enough empty time before a tape stop.

Course note-offs do not damp free courses.
Use the chanter's CC64 to stop them.
Do not use `let_ring` on the skinpipe or skinthumb: their shared damper is an instrument control, not harmonic-boundary sustain.
Scored CC64 survives the performance pass for this model.
A new thumb pluck replaces that course's stored wave; the other courses and the shared head retain their histories.

Score-addressed controls work in the ordinary performance sidecar:

```toml
[[controls]]
track = "skinpipe"
bar = 3
beat = 1
control = 24
value = 0
```

## Bloom and halo

D4 and A3 strongly wake the matched courses; held E4 leaves less total resonance.
The acceptance test requires at least 10 dB more summed course energy for D4 than E4, a 3–8-second halo T60 and a halo at +1 second no lower than −30 dB relative to pre-cut sound.
The radiated skin path is approximately level with dry radiation on held D4, giving the lean gesture a larger share of the sound.

**Pressure and resonance are coupled, not interchangeable.**
The courses retain fixed tuning and long decay.
The closed-head reed has lower pressure sensitivity and a bounded reactive head load that pulls a nearby pitch toward resonance.
The acceptance test checks every integer CC2 from 90 through 108 against a ±3-dB course-energy criterion relative to 96.
Pressure need not increase bloom monotonically, and 112 is outside that interval.
A conservative intimate phrase can use 92–100.
The head locks pitch only over a narrow central range and merely attracts it at greater detuning; it does not capture a full ±10 cents.

E4 is not entirely dry: its second harmonic matches the A courses' third harmonic in this just tuning.
The physical partial coincidence leaves an A-family halo even when the D courses stay quiet.
No pitch-class switch or generated pad removes it.

The body responds to its own reed and thumb only.
Other band stems, including banjo and hum, do not acoustically drive its chamber or courses.

## Tests

`tests/test_skinpipe.py` measures the acceptance criteria above: pitch, course bloom, halo decay, pressure response, plucks and damper.
