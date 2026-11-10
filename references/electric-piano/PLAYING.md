# Tine electric piano: playing sheet

Use `model = "electric-piano"`; MIDI **28–100 (E1–E7)**, the 73-key range.
There are no keyswitches, program-change presets or track-name gestures.
Every note needs a note-off.
Velocity continuously changes hammer impulse, modal brightness and nonlinear pickup bark; there are no sample-layer boundaries.
CC64 sustain is independent on each MIDI channel.

```toml
[instruments]
keys = { model = "electric-piano", gain_db = 0, pan = 0 }

[electric_piano_model]
preset = "tremolo"

[[controls]]
track = "keys"
bar = 1
beat = 1
control = 92
value = 64
```

`[[controls]]` uses one-based score bar/beat positions and follows the performance clock.
Generators can emit ordinary channel-matched MIDI CC events instead.
LilyPond `\sustainOn` / `\sustainOff` emits the supported CC64 messages.
Set the effect mode before the note-on or in a rest.

| Gesture | MIDI request | Behavior |
| --- | --- | --- |
| Soft / singing / barking strike | Velocities about 35–60 / 60–90 / 95–127 | Continuous hammer/pickup response; use the `bark` preset for much stronger high-velocity bite. |
| Damper release | Note-off, CC64 up | Felt damping has 55 ms amplitude e-fold time, plus a low-frequency felt thump. No pedal means damping begins at note-off. |
| Sustain | CC64 ≥64 down, <64 up | Holds released notes until pedal-up. No half-pedal or sympathetic resonance. Close the final pedal explicitly. |
| Swell / volume | CC11 0–127; default 127 | 10 ms smoothing, volume only; unlike velocity it doesn't change the strike. |
| Effect off / tremolo / stereo vibrato | CC92 **0 / 64 / 127** | Tremolo modulates both channels together. “Stereo vibrato” is the Rhodes suitcase-style **opposed amplitude modulation**, not pitch vibrato. |
| Effect depth | CC93 0–127; preset default about 64 | Depth 0–1, 20 ms smoothing. At full depth each channel ranges from silence to unity. |
| Effect rate | CC94 0–127; default about 53 | **0.5 + 9.5×value/127 Hz**, 20 ms smoothing. CC94=53 is about 4.46 Hz; rate is not tempo-synchronized. |
| Pickup position / amp drive | Preset TOML | `pickup.offset`, `gap`, `excursion` and `amp.drive`; these aren't MIDI CC gestures. |

A pedal held to the end sustains to the render boundary and can produce a cut tail.
Write the final CC64=0 early enough for at least 0.7 s of release; the renderer adds 2 s after the MIDI's end.
Pitch bend, CC1, CC67 and release velocity have no effect.
Retriggering a held pitch starts another independently decaying modal voice; repeated pedal-down notes accumulate rather than choking one another.

## Presets and timbre choice

| Preset | Use |
| --- | --- |
| `default` | Rounded proxy-fit pickup/decay, effects off. Fit uses two Mark II notes, **not a verified Mark I**. |
| `tremolo` | Same pickup, 4.5 Hz tremolo at depth 0.5. |
| `bark` | Stronger pickup curvature, bell modes and coupled-tonebar beating; measured long decay retained. An expressive alternative, not the best reference fit. |

The rounded and bark presets are calibrated toward −24 LUFS on the same ballad passage before mixing.
Choose by ear; the lower fitted loss doesn't settle the composer's timbre choice.

## Useful writing

For a soul ballad, use left-hand roots/octaves around E2–C3, open seventh/ninth voicings around C4–C5, and velocities 60–85.
Pedal each harmony, then clear just before the next root; slow tremolo at depth 0.3–0.55 leaves the phrasing visible.
For gospel-rock fills, use short mixolydian/blues fragments, sixths and clipped dominant voicings around A3–E5, velocities 85–110, with little pedal.
Dense close-position bass chords muddy the pickup; space the bass and keep most chord tones above middle C.
Very long held treble notes extrapolate far beyond the two reference pitches; the modal T60 is bounded to 30 s, but the extrapolated color isn't validated.
Large pedal-down clusters can still clip; the renderer reports clipping rather than normalizing notes or hiding it with a limiter.
