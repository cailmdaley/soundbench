# Electric bass — playing sheet

Use sounding pitches, not the bass-clef octave convention.
The model is a four-string electric bass, with explicit note lengths and no samples.

```toml
[instruments]
bass = { model = "electric-bass", gain_db = 0, pan = 0 }

[electric_bass_model]
preset = "round"                      # flatwounds by default; or "warm"
```

## Notes and gestures

| Gesture | MIDI instruction | Effect / limits |
| --- | --- | --- |
| Finger pluck | Note-on; CC22 = 0 | Default; velocity 55–110 is useful. |
| Pick | CC22 ≥ 64 before note-on | Shorter, brighter excitation. |
| Staccato / fretting-hand release | Short note, ordinary note-off | Every string, including an open string, damps with 80-ms T60. |
| Palm mute | CC21 ≥ 64 before note-on | Pluck followed by rapid damping; switch CC21 back to 0 for normal notes. |
| Ghost / dead stroke | CC20 ≥ 64 before note-on | Quiet seeded contact noise and a very short pitched residue; CC20 takes priority over CC21. |
| Explicit string | CC23 = 1, 2, 3, 4; 0 = automatic | High to low: G2 = 43, D2 = 38, A1 = 33, E1 = 28; 24 frets. |
| Slide / bend | MIDI pitch wheel, ±12 semitones | Smooth 10-ms retuning of held strings on the same channel; return wheel to 0. |
| Hammer-on / pull-off | Pin CC23, hold source note, set CC68 ≥ 64, play destination | Higher = hammer; lower = pull; preserves the source wave and adds a small transient. |
| Hand choke | CC123 or CC120 | Damps strings owned by that MIDI track/channel. |

Controllers latch separately for each MIDI track/channel.
CC68 must be 0 before an ordinary pluck, even if a ghost or mute flag is set.
A legato destination needs a held source on the same pinned physical string; an unpinned or unplayable gesture fails.
Release the source after the destination begins, then release the destination normally.
A source note-off cannot damp its replacement because events carry unique note tokens.

Pitch wheel range is fixed at ±12 semitones in this renderer.
For portable MIDI, emit RPN 0 with data-entry MSB 12, LSB 0; the renderer does not interpret alternate RPN ranges.
Wheel changes affect every held string on that channel, so use separate channels for independent bends.
CC64 ≥ 64 is rejected: sustain pedal is not an electric-bass articulation.
Every note must have a note-off; unterminated MIDI fails.
LilyPond emits pitches, durations and velocity, but these CC gestures need a MIDI generator or a post-processing pass.
No score-sidecar electric-bass gesture table is implemented.

## Useful range and timing

The supported range is E1–G4, MIDI 28–67.
Keep most funk lines in E1–E3; the upper frets get very thin.
Use 70–180 ms notes for dry stabs and 300–900 ms for held answers.
Leave an actual gap before the next attack if you want the note release to be audible.
A ghost still uses a playable scored pitch; don't use a keyswitch outside the range.
Use velocity 35–75 for ghosts, rather than velocity 1.

Automatic fingering chooses the lowest playable fret, not a hand-position model.
Pin a string for consistent pickup color or legato.
This is one shared physical instrument across its tracks; competing notes on the same string replace one another.
Palm mute and ghosts are signal-model approximations, not measured bridge/hand impedances.
The P-style flat preset and two-pickup J-style round preset are constructed alternatives, not authenticated historical gear.

## Generator API

`soundbench.scripts.models.electric_bass` exports `note`, `assign_string`, `load_config`, and `render`.
`note(on, off, pitch, velocity, string, token, kind=0, hardness=0, owner=0)` takes seconds and a one-based string.
Kinds are 0 pluck, 3 hammer, 4 pull, 6 ghost, 7 palm-muted pluck; hardness is 0 finger or 1 pick.
Use a unique nonnegative integer token for each note.
The 11-column shared-waveguide event format also supports kind 2 pitch ramps and kind 5 token-addressed chokes.
`uv run python -m soundbench.scripts.electric_bass_lab probes --out <dir>` renders an example of every gesture.
