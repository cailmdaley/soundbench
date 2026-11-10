# Flabiol and xeremia — composer sheet

Bind any stable LilyPond Voice/Staff track name in the performance sidecar:

```toml
[instruments]
flabiol = { model = "flabiol", gain_db = 0, pan = -0.2 }
xeremia = { model = "xeremia", gain_db = 0, pan = 0.2 }
# Optional preset (default, d, g, or a file of your own) and static gain:
[flabiol_model]
preset = "d"
output_gain = 0.37
[xeremia_model]
preset = "g"
```

**Write sounding pitch; no keyswitches and no automatic transposition.**
Track names only route the instrument; all controls must be on its own note-bearing MIDI track.
Both winds are monophonic, last onset wins across overlapping notes; simultaneous chords fail.
Use one track per wind player.
The renderer rejects missing note-offs, out-of-range notes and nonpositive durations.
CC64/pedal does nothing.

| Instrument | Accepted MIDI pitches | Practical guidance |
| --- | --- | --- |
| Flabiol | 74–98 (D5–D7) | Prefer C6–C7 alongside C4–C5 grall; this is a model range, not an authenticated Mallorcan fingering map. |
| Xeremia grall | 59–72 (B3–C5) | Prefer C-major naturals; B-flat/F-sharp are possible idiomatic alterations. Other chromatic steps are diagnostic conveniences. |
| Bordó | CC21 = 36–60 (C2–C4) | Default C3 = 48; normally leave fixed for the piece. One sounding drone, no artificial audio for decorative fillols. |

Both presets use the same constructed C-relative just offsets.
Those offsets are editable in `tuning_cents` (12 entries, C through B); they are not measured Mallorcan tuning.
A transposed instrument needs a correspondingly rotated tuning table and adjusted drone.
A piece-scoped `pitch_range = [low, high]` declares a differently sized chanter's sounding compass; the effective bore delay follows each sounding frequency. The default remains B3–C5. G4-tonic plagal chanters may use D4–E5, and D4-tonic ones A3–B4, without octave folding. Phase correction is extrapolated, not remeasured.
Velocity 55–110 is useful for the flute; 16–127 is tested for smooth amplitude response.
Grall velocity is only a gentle effort/loudness change: it never tongues the reed.

## Gestures

| Score request | MIDI / generator request | Result |
| --- | --- | --- |
| Tongued flute note | Ordinary note-on/off, CC68 = 0 (default) | 18-ms breath attack with a roughly 15-ms decaying chiff, 100-ms finite release; every onset owns the new envelope. |
| Flute slur | CC68 ≥ 64 before notes; successive note gap ≤25 ms | Retains bore history and breath; hard fingering change, not a portamento. Set CC68 = 0 to resume tonguing. LilyPond slur marks alone emit no CC68. |
| Mordent / turn / reguinyol | Explicit `\grace { d''32 c'' }` etc., in the chosen sounding octave | LilyPond emits short notes; renderer treats them as fingering changes. Inspect the resulting MIDI timing. |
| Trill | Write the alternating pitches and durations; e.g. 30–70 ms per flute note | `\trill` alone produces no alternating MIDI notes. No automatic ornament generator. |
| Flute breath accent | Raise velocity at onset or send CC2 = 105–115 briefly | More breath; extreme pressure can change oscillation regime. |
| Written dynamic expression | CC11: 0 silent, 127 full | A 12-Hz-smoothed radiated-level envelope before room/tape; CC2 independently drives the physical jet/bag. Useful for phrase and section arcs without pushing the reed into an unstable pressure regime. |
| Breath / bag pressure | CC2: default 96; 0 = no drive, 96 = nominal, 127 = 1.323× nominal | Sample-smoothed physical drive; for xeremia the same pressure feeds drone and grall and shifts their pitch (180 cents per unit relative pressure, e.g. CC2 96→102 gives +11.25 cents). Stable nominal region is roughly 85–110; low pressure can stop or squeak. |
| Vibrato | CC1: 0 off (default), 127 full | Flute ±18 cents, grall ±8 cents at 5.4 Hz; with CC1 at 0 the flute's breath still flutters by about 3 cents at 1.5–6 Hz (`tremor_cents`, 0 for a machine-steady jet); the bag's separate squeeze/refill pressure wander is shared by both pipes, up to ±4.5 cents with 2–4-s control intervals, plus a 1.5–6 Hz arm flutter of about 3 cents (`tremor_cents`, at most 10). |
| Drone off/on | CC20: 0 off; 127 on (default); intermediate values scale its pressure | Separate reed/bore; switching off releases stored drone energy, not a hard audio mute. |
| Drone pitch | CC21 = MIDI pitch, default 48 | Frequency uses the same tuning table. Do not write a simultaneous low note/chord in the grall track. |
| Start squawk | Automatic at first scored onset; CC22 = 0 suppresses, 127 full (default) | 750-ms low-pressure rise: both pipes start up to 240 cents flat and wobble upward, with reed turbulence and escaping-air noise. CC22 = 0 removes that bend/noise but retains the pressure rise. This is designed, not fitted to an onset recording. |
| End sag | Automatic after last note-off; CC23 = 0 suppresses, 127 full (default) | 550-ms pressure fall with up-to-190-cent droop, then bore decay. Leave at least 1.2 s tail. |
| Chanter off/on | CC25: 0 off, 127 on (default) | Independent reed gate; the drone and bag keep sounding. A scored note is still needed to establish the bag's lifetime. |
| Mid-piece bag stop/restart | CC24 = 0 stops bag, CC24 = 127 resumes | Off/on ramps and sag/squawk within the first-onset/last-off window. Normal scored rests **do not** stop the bag. |

For xeremia, the bag starts at the first note and stops at the **latest scored note-off**.
Through rests it keeps the last fingering, so a repeated note is parted by a cut: the engine starts it with 30 ms (at most half the note) of the note a whole tone above, or a whole tone below at the top of the compass (`repeat_cut_ms`; 0 disables).
Write your own grace instead when you want a different cut (a G4 cut between repeated C4s is the classic one); a written grace is a different pitch, so no automatic cut is added.
The reed itself sounds raw: flow-driven turbulence gives it a recorded bagpipe's noise floor, and open-hole radiation (`radiation`) gives the reedy 2–4 kHz edge. Piece presets inherit these unless they set them.
Flute turns/mordents often work at 35–80 ms per grace; extremely short (<15-ms) tongued notes can become mostly breath transients.
Grace examples must be kept inside each model's range.

LilyPond supplies grace-note pitches/timing and dynamics; arbitrary CCs require MIDI postprocessing or the generator layer.
Inject a `mido.Message('control_change', control=2, value=105, channel=...)` at the intended tick on the named track, preserving sorted delta times.
The engine API accepts `render(rows, seconds, config, instrument, controls)` with notes `{time, off, pitch, velocity}` and controls `{time, control, value}` in seconds.
No sidecar `[[articulation]]` extension is needed: ornaments are notes and breath gestures are CCs.
