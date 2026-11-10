# Playing the fiddle / violí

Use `model = "fiddle"` for either Americana fiddle or Mallorcan violí.
Track names are ordinary selectors, not articulations: `fiddle`, `violi`, `drone` and any stable Voice name work.
There are no keyswitches, sustain-pedal behavior or special percussion-note aliases.
Every MIDI note needs a note-off.

```toml
[instruments]
fiddle = { model = "fiddle", gain_db = 6, pan = 0.15 }

[fiddle_model]
tuning = "AEAE"          # exactly standard, AEAE, ADAE, GDGD
# preset = "modern"   # brighter, untilted; the default is the old-time fiddle

[fiddle_strings]
fiddle = 2               # optional selector -> physical string, high to low
# drone = 4
```

`gain_db = 6` is a starting mix setting, not a calibrated balance against every other instrument.
A raw old-time reel stem measures about −36 LUFS before stem gain, room and mastering.
The default preset is the old-time tone (−1 dB/octave above 1 kHz); `modern` is the same body untilted, as a close modern recording hears it.

## Pitch, dynamics and physical strings

| String | Standard open pitch | AEAE | GDGD |
|---|---|---|---|
| 1, highest | E5 / MIDI 76 | E5 / 76 | D5 / 74 |
| 2 | A4 / 69 | A4 / 69 | G4 / 67 |
| 3 | D4 / 62 | E4 / 64 | D4 / 62 |
| 4, lowest | G3 / 55 | A3 / 57 | G3 / 55 |

Each string accepts its open pitch through open +24 semitones.
The tested chromatic range is MIDI 55–96; high notes and extreme controls aren't a promise of violin realism.
Automatic allocation picks the lowest finger position on a free playable string.
Pin strings for slurs, double stops and drones; there is no hidden voice stealing.
Overlapping notes on the same pinned string fail, as do notes below that string's open pitch.
Practical fiddle melody: MIDI 69–81, usually string 2 for a consistent slurred phrase.
Old-time velocities 70–105 work well; velocity 1–127 is supported.
Velocity sets intensity; the player supplies bow-speed swells, bow-change dips and an inferred phrase-end decrescendo.
Normal notes get pitch-dependent force against speed and position, then a mechanical-state preview chooses a rich, in-tune setting.
Explicit CC2 / generator `force` values or force events make that entire note manual and bypass the tone-preservation guard.
Use CC22 for deliberately normal, surface, crunchy or raw bowing.
Grit (CC27, or `grit` 0–1) is the tone colour, from sweet (0: light grain, bow further from the bridge) through the house tone (0.25) to gritty (1: double grain, nearer the bridge). Old-time reels sit around 0.35; an air around 0.1–0.25.

## Channel-scoped MIDI controls

Set controllers before note-on; CC2, CC4 and CC8 also update every held note on that channel.
Use separate channels for independently controlled drone and melody voices.
CCs persist until changed; separate files don't inherit controller state.

| CC | Meaning and mapping | Starting value |
|---|---|---|
| 2 | Manual bow normal force (bypasses tone guard for the note): `0.002 + 0.998 × value/127` newtons | 15 ≈0.12 N on strings 1–2; 20 ≈0.16 N on 3; 25 ≈0.20 N on 4 |
| 4 | Bow speed magnitude: `0.01 + 0.49 × value/127` m/s | 28 ≈0.118 m/s |
| 8 | Bow position β from bridge / sounding length: `0.04 + 0.26 × value/127` | 39 ≈0.12 |
| 1 | Finger vibrato depth: `35 × value/127` cents | 0 off; 54 ≈15 cents for an air |
| 5 | Slide time: `0.01 + 0.99 × value/127` seconds | 14 ≈0.12 s |
| 16 | Upward slide into the scored note: starts `−3 × value/127` semitones below it | 0 off; 42 ≈1 semitone |
| 20 | Physical string: 0 auto; 1–4 high to low | 0; pin for gestures |
| 21 | Chop approximation: 0–63 ordinary, 64–127 short pressure pulse | 0 |
| 68 | Slurred destination: 0–63 new bow, 64–127 same bow as preceding note on that string | 0 |
| 22 | Bow style: 0 normal tone guard, 1 surface, 2 crunch, 3 raw / no player shaping | 0 |
| 23 | Phrase ending: 0 infer from next note, 1 decrescendo, 2 continue | 0 |
| 25 | Vibrato delay: `2 × value/127` seconds | 16 ≈0.25 s |
| 26 | Vibrato rate: `4 + 4 × value/127` Hz | 48 ≈5.5 Hz |
| 27 | Grit, sweet to gritty: `value/127` | 32 ≈0.25 (house tone) |

Absent physical CCs, commanded speed is 0.12 m/s and β=0.12; actual speed swells from about 84% to 112% before bow-change/phrase-end dips.
Force is chosen per pitch and mechanical history, not fixed by velocity or string number.
Normal stopped notes longer than 0.7 s default to 12-cent vibrato after 250 ms, ramping in over 120 ms at 5.5 Hz; open strings ignore it.
Set CC1=0 for a deliberately straight long tone.
CC2 supplies absolute manual newtons, including for notes carrying mid-note force events; raw mode uses preset force until overridden.
CC1/5/16/20/21/22/23/25/26/27/68 are latched at note-on, not edited mid-note.
CC64 and MIDI pitch bend are ignored.

## Gestures

| Gesture | Exact request | Advice |
|---|---|---|
| Détaché / bow change | CC68=0 before destination note-on; separate notes with explicit note-offs | Bow direction alternates on that string with a short lift; 0.2–0.6 s notes are useful. |
| Slur | Destination starts exactly at preceding note-off; CC68=127 before destination note-on | On one string, bow direction and waveguide state persist. Onto another string, the stroke keeps its direction and rolls across in about 8 ms. Set CC68=0 at the next new bow. Gaps over 15 ms cancel the slur. |
| Nashville shuffle | Equal eighth notes, repeat bow groups **2–1–1**: CC68 sequence **0,127,0,0** | Accent the first eighth (velocity ≈100 vs 85). Repeat gives 2–1–1–2–1–1 over eight eighth notes. |
| Georgia shuffle | Equal eighth notes, repeat bow groups **3–1**: CC68 sequence **0,127,127,0** | Accent the isolated fourth eighth, not every quarter. Phase the group against the tune deliberately. |
| Double stop | Two simultaneous pitched notes on distinct pinned strings/channels | Bow them together; independently chosen forces are an approximation to one bow contacting both strings. Avoid triple/quadruple stops. |
| Open drone | Hold the desired tuning's open MIDI note on a separate pinned string | A sounding drone needs a scored note; passive sympathetic ringing is quieter and automatic. Start drone notes with the melody stroke they ride: the instrument has one bow, so strokes starting together share a direction. |
| Round Peak bowing | `soundbench.scripts.models.fiddle_player.oldtime(melody, beat, open_midi)` | Turns a list of eighths into off-beat slurs, a same-bow drone on the adjacent string, 3+3+2 rocking swells, slides into chosen pitch classes and pushed off-beats. |
| Slide | CC16>0 and CC5 before note-on | Destination is the MIDI pitch; don't start below the open string. Reset CC16=0 before ordinary notes. |
| Finger vibrato | CC1 at note-on | 0–8 cents suits a restrained reel; 10–20 cents after 250 ms suits the air. Open strings ignore finger vibrato. |
| Cross-tuning | `fiddle_model.tuning` in sidecar | Pitches are absolute MIDI, not transposed finger positions. Retuning mid-render isn't supported. |
| Chop | CC21=127, velocity 70–100, duration ≈0.08–0.12 s; CC2≈57, CC4≈13, CC8≈78 | Approximate friction pulse, not a measured chop; reset CC21 and physical CCs afterward. |
| Surface / crunch | Change CC2 while keeping speed and position fixed | At A4, ≈0.12 N sustains harmonics; ≈0.35–0.8 N becomes noisy/crunchy. Use CC22=1/2 to bypass normal tone preservation. Very low forces can still be almost silent rather than useful surface sound. |

LilyPond's printed slurs do **not** emit these controllers.
Use the generator layer or explicitly inserted MIDI CCs for bowing; don't assume engraving translates into performance.
Track names don't fix that limitation.

## Generator API

```python
from soundbench.scripts.models.fiddle import load_config, render

rows = [
    dict(time=0, off=.5, pitch=71, velocity=95, string=2,
         slide=-1, slide_seconds=.15),
    dict(time=.5, off=1., pitch=73, velocity=85, string=2, slur=True),
    dict(time=0, off=1., pitch=57, velocity=55, string=4),
]
audio = render(rows, 3., config=load_config(tuning="AEAE"))[0]
```

Required note keys: `time`, `off` in seconds, absolute `pitch`, `velocity`.
Optional: `string` 1–4, `owner` zero-based stem index, `slur`, `force` .002–1 N, `speed` .01–.5 m/s, `position` .04–.3, `vibrato` 0–35 cents, `vibrato_delay` 0–2 s (default .25), `vibrato_rate` 4–8 Hz (default 5.5), `phrase_end` boolean (absent = infer), `bow_style` normal/surface/crunch/raw, `player` boolean, `slide` −3..+3 semitones relative to destination, `slide_seconds` .01–1, `grit` 0–1 (default .25), and `chop`.
Positive `slide` starts above the note and slides down; MIDI CC16 exposes upward slides only.
For held controls, `controls=[(seconds_since_onset, "force"|"speed"|"position", value), ...]` uses physical units.
Supply `seconds` long enough to include the final note-off plus two seconds of tail.
Stopped notes retain 55 ms of free ringing after bow lift before finger damping; open strings retain natural decay.
For raw-mechanics research, `render(..., player=False, noise=False)` disables performance shaping and the rosin grain in the friction.
The normal player uses deterministic offline rehearsal (7 force factors per held interval, judged by Helmholtz motion), so rendering is more expensive than the bare waveguide; a 255-s three-track reel renders in about 200 s. `render(..., channels=1)` returns one channel per owner.
