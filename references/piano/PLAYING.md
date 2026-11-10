# Playing the modelled grand

Middle C is C4 = MIDI 60.
Use plain MIDI; there are no keyswitches or track-name articulations.

```toml
[instruments]
piano = { model = "piano", gain_db = 0, pan = 0 }

# Optional, piece-wide:
[piano_model]
preset = "default"                # or "close"
```

The piano binding inherits the grand-piano stereo balance and automatic-pedal handling.
The model needs ensemble-style `[instruments]` routing even for solo piano.

| MIDI gesture | Result |
| --- | --- |
| Note-on, MIDI 21–108, velocity 1–127 | Sounding piano pitch, equal-tempered first partial, fitted stiff upper partials |
| Note-off / note-on velocity 0 | Release the key; every final note needs an explicit release |
| CC64 = 0 | Damp released notes; nominal T60 of added damper loss is 0.22 s |
| CC64 = 1–126 | Continuous partial damping; 32, 64 and 96 are useful half-pedal settings |
| CC64 = 127 | Released keys retain natural decay; chromatic sympathetic resonators are enabled |
| CC67 = 0–127 | Continuous soft-pedal approximation: quieter and darker; not a fitted hammer-shift mechanism |

Controllers are channel-local, latch until changed, and should be sent before the affected note.
CC64 can change while a released note rings, including lifting and re-pedalling.
Repedalling stops further damping but cannot restore energy already lost.
The highest notes, MIDI 90–108, have no key damper; they decay naturally even at CC64 = 0.
MIDI pitch bend, sostenuto, release velocity and aftertouch are not implemented.

## Writing range and touch

Start in **MIDI 36–84 (C2–C6), velocities 30–80**.
The fit covers all 88 keys, but the extremes and fortissimo are less trustworthy.
These dynamic labels are writing guides, not calibrated acoustic SPL:

| Velocity | Suggested dynamic |
| --- | --- |
| 20–35 | pp |
| 36–50 | p |
| 51–65 | mp |
| 66–80 | mf |
| 81–100 | f, provisional |
| 101–127 | ff, provisional; leave mix headroom |

Velocity interpolates continuously between fitted layers; it does not switch samples.
The middle register suits slow melody, broken chords and modest chordal accompaniment.
Avoid exposed high-treble pianissimo, percussive repeated ff chords, or relying on una corda to change the number of struck strings.
The bridge-coupled model has fitted aftersound and a sample-free body/key-bottom layer.
Residual spatial phase and the highest-octave dispersion remain approximate.

A voice is retired with a 100-ms fade after at most 18 seconds.
The ordinary render tail is four seconds, so explicitly lift CC64 at the end of a passage.
The output has static gain and no limiter; dense loud chords can still exceed full scale.
