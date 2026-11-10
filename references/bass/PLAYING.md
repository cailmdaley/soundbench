# Playing the modelled upright bass

Use sounding pitches, not the octave-transposed bass notation pitch.
The model defaults to thumpy gut-like pizzicato; `steel` is the longer-decaying alternative.
Velocity changes attack hardness continuously as well as loudness; 45–65 is softer, 85–105 speaks more strongly in the band.
No samples, keyswitch notes or GM programs select gestures.

```toml
[instruments]
bass = { model = "bass", gain_db = 1, pan = 0 }

[bass_model]
preset = "gut"                   # gut or steel
intonation_cents = 2.0            # seeded stopped-note variance, 0..15 cents
# parameters = "instruments/models/bass-gut.toml"
# output_gain = 1.0               # absolute engine gain, not dB; omit for calibration
# seed = 1892

[bass_strings]
bass = 3                         # optional track pin: 1 G2, 2 D2, 3 A1, 4 E1

[[bass_articulation]]
track = "bass"
bar = 5
beat = 1
kind = "slide"
string = 3
to = 5                          # relative semitones; fractional values work
duration_beats = 1.0             # finish before source note-off; tie/lengthen it
shape = "ease"                   # ease (default) or linear
```

`track` accepts the full MIDI track name or a colon component, e.g. `bass` for `bass:bass`.
A gesture must select exactly one bass track.
Bars and beats are one-based, in the score's meter; durations use meter beat units.
The renderer anchors gestures to performed notes, including groove and rubato.
One gesture per score onset; all attacks except `damp` require exactly one note there.
Automatic string choice prefers the highest free playable open string; pin a string to retain its color or to slide below that default string's open pitch.
Four tracks still share four physical strings; a new attack replaces that string's existing wave.

| Request (`kind`) | Extra fields | Result and useful range |
|---|---|---|
| Ordinary MIDI note / `pizz` | `string`, `position`, `cents` optional | Flesh pluck with a brief bright release front and point-shaped contact transient; velocities 65–105 for a folk line. MIDI 28–55 is the comfortable register; 28–67 is accepted. |
| `pizz`, different point | `position = 0.02..0.48` | Fraction of sounding length from bridge. 0.07 bridge: lean/bright; 0.22 ordinary; 0.40 over fingerboard: round. Zero/omitted uses preset. |
| `slap` | Optional `position`, `cents`, `string` | Large displacement hits a unilateral fingerboard barrier. Use 85–110 for pitched snap/slap. This retriggers the scored pitch; it isn't a separate unpitched keyswitch. |
| `slap-back` | Optional fields as for slap | Same collision with fast left-hand damping. Use 65–85 on beats 2/4, scoring the preceding root/fifth pitch with a short gate. Below about 50, the displacement may not reach the barrier. |
| `dead` | Optional fields as for pizz | Strongly damped string plus contact thump. Write the actual string's pitch, velocity 60–100; note-off still required. |
| `ghost` | Optional fields as for pizz | Reduced displacement with the dead-note decay; 25–60 for quiet offbeat thumps. Body/contact color remains. |
| `slide` | Required `to`, `duration_beats`; optional `shape`, `string`, `position`, `cents` | Plucks the scored starting pitch, then continuously retunes that same string. Destination must remain on the string and slide must finish before release. No second attack at destination. |
| `damp` | Optional `string` | Damps preceding notes at the addressed beat; no scored attack needed. Without `string`, damps each string previously used on that track. |
| Slight intonation | `cents = -50..50` on an attack | Offset in cents, plus the preset's seeded stopped-note variance. Open strings have no automatic variance. |

Add each gesture as its own `[[bass_articulation]]` table.
`kind = "pizz"` is also the way to pin just one note's string or position.
Track pins and per-gesture `string` are **one-based**; engine events use zero-based strings.
No fret quantization is applied to fractional MIDI pitches in the generator API.

**Two-feel:** write root on beat 1 and fifth on beat 3, root velocity about 98 and fifth about 88.
At 109 bpm, let each note last about 0.62 seconds rather than a whole half note; leave air before the next attack.
For walking quarters, use about 0.40-second gates, velocities 84–100, and approach the next root by a semitone.
For bluegrass slap-back, add a same-pitch `slap-back` on beats 2 and 4, shorter gates and lower velocity than the roots.
Do not expect printed staccato or slurs alone to change LilyPond MIDI gates.
Write the durations or use the performance layer.

MIDI note-off damps **every** string, including opens; CC64 extends the gate until pedal-up.
A later note on the same string is safe from the earlier note's stale note-off.
Pitch-wheel, other CCs, track-name articulation suffixes, keyswitches, hammer/pull and arco are not implemented.
Avoid four-note fortissimo slap chords: the output safety curve keeps them finite, but they are neither idiomatic nor a clean dynamics test.

## Generator API

Import `soundbench.scripts.models.bass`.
`note(on, off, pitch, velocity, string, token, owner=0, kind="pizz", position=0, cents=0)` returns an attack and release in seconds converted to 48-kHz sample indices.
Tokens must be unique across voices.
`render(events, seconds, owners=1, config=load_config())` returns one mono stem per owner.
Each row is `[sample, kind, string0, owner, pitch, velocity, ramp_samples, shape, position, cents, token]`.
Kinds are `0 pizz, 1 release, 2 slide, 3 slap, 4 dead, 5 ghost, 6 damp, 7 slap-back`; shape is `0 linear, 1 ease`.
A slide row uses the **absolute destination** MIDI pitch, unlike the sidecar's relative `to`.
Sort same-sample releases before replacement attacks and slide rows after their source attack.
