# Tamborí: score contract

```toml
[instruments]
tambori = { model = "tambori", gain_db = 0, pan = -0.2 }
```

Use a named pitched MIDI Voice (`tambori`), **not** LilyPond `\drummode` with a GM drum map.
All keys select gestures; this instrument is not chromatically pitched.
There are no keyswitches, pitch bends, CC64 sustain or track-name articulation suffixes.
Track names only route the binding.

| MIDI | LilyPond absolute pitch | Gesture | Note length |
| --- | --- | --- | --- |
| 36 (C2) | `c,` | Center stroke | Ignored after the hit |
| 37 (C♯2) | `cis,` | Edge stroke at 72% of radius | Ignored after the hit |
| 38 (D2) | `d,` | Accented center stroke, ×1.25 amplitude | Ignored after the hit |
| 39 (D♯2) | `dis,` | Flam: soft stroke at onset, main stroke 27 ms later | Ignored; both strokes play |
| 40 (E2) | `e,` | Bouncing roll: hits every 34 ms, four-hit decaying groups | Stops generating hits at note-off |
| 41 (F2) | `f,` | Rimshot: edge head + shell | Ignored after the hit |
| 42 (F♯2) | `fis,` | Shell click only | Ignored after the hit |

Supply an explicit note-off for **every** note, including hits; unclosed notes fail rendering.
Natural decay rings after note-off; note-off is not a choke.
Velocity 1–127 is continuous; 45–85 suits accompaniment, 90–110 an accent.
Avoid overlapping velocity-127 rolls/rimshots: the renderer deliberately rejects clipping instead of normalizing it away.

For a simple triple-meter exercise at quarter=160, use D2 on beat 1, C♯2 on beat 2, and C2 on beat 3, adding quiet offbeat strokes sparingly.
For a binary toc, alternate a strong center and lighter edge stroke on quarter/eighth subdivisions.
These are demonstration patterns, not transcriptions of a named traditional toc.
The flam's main beat falls **27 ms after** the MIDI onset; place the MIDI onset early if the main stroke must land on a beat.
Rimshots/shell clicks are model extensions; their traditional local usage isn't established by the research.

The modest 139 Hz body mode is an unfitted shell-breathing prior, not a sample or a high-pass bypass.
