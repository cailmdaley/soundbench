# Castanyoles: score contract

```toml
[instruments]
dancers = { model = "castanyoles", gain_db = 0, pan = 0.25 }

[castanyoles_model]
parameters = "instruments/models/castanyoles-fitted.toml"
dancers = 6
timing_ms = 8
tuning_cents = 16
```

The fitted preset is the brighter recording-fit alternative.
Omit `parameters` for the physical-prior preset.
Use a named pitched MIDI Voice (`dancers` here); track names route bindings and do not select hand or gesture.
No GM drum map, keyswitches, pitch bends or CCs are used.
All keys select gestures, not a chromatic scale.

| MIDI | LilyPond absolute pitch | Gesture | Note length |
| --- | --- | --- | --- |
| 60 (C4) | `c'` | Femella/right: single click | Ignored after hit |
| 61 (C♯4) | `cis'` | Mascle/left: lower single click | Ignored after hit |
| 62 (D4) | `d'` | Right multi-finger roll, hits every 42 ms | Stops creating clicks at note-off |
| 63 (D♯4) | `dis'` | Left roll (convenience, not a traditional gesture) | Stops creating clicks at note-off |
| 64 (E4) | `e'` | Golpe: both hands together | Ignored after hit |

Roll velocity repeats `[1.0, 0.8, 0.7, 0.88]` relative to the scored velocity.
A four-finger group lasts about 0.17 s; for exact finger timing write individual C4 clicks rather than a roll trigger.
Velocity 1–127 is continuous; 45–85 works well, 90–110 marks a cadence.
Every note needs an explicit note-off, even a single click.
Natural decay rings after note-off; no sustain/choke control is implemented.
Avoid chords of multiple trigger keys when one E4 golpe suffices; duplicated pairs can clip at high velocity.

`dancers` is 1–16 **pairs**, with one persistent tuning offset and timing bias per pair plus per-click timing jitter.
The engine scales a group by `1/sqrt(N)` so adding dancers changes texture without multiplying loudness by N.
`timing_ms` (0–30) is bias standard deviation; per-hit jitter uses half that value.
`tuning_cents` (0–100) is pair detuning standard deviation; the mascle remains ×0.87 in frequency relative to its femella.
Those hand labels and that interval are composer conventions, not measured Mallorcan tuning.
Use 4–8 pairs with 5–10 ms timing spread; 16 near-unison pairs or zero spread sounds unnaturally aligned.
The ensemble is mono within its routed stem, not one stereo seat per dancer.
