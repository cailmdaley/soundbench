# Drum machines: playing sheet

Three machines, played like any other model stem: a circuit-modelled **TR-808** (`tr808`), a **TR-909** (`tr909`, analog drums plus 6-bit "virtual ROM" cymbals) and a **Linn**-style ROM machine (`linn`, the library's own physical drum kit frozen into 8-bit companded ROMs).
The 808's bass drum also plays as a pitched, gliding **808 bass** (`tr808-bass`).
Each renders a fixed-gain stereo stem at 48 kHz.

```toml
[instruments]
kit  = { model = "tr808", gain_db = 0, pan = 0 }       # or "tr909" / "linn"
boom = { model = "tr808-bass", gain_db = -3, pan = 0 } # optional pitched 808 on its own track

[tr808_model]               # the box; also governs tr808-bass
unit = "studio-a"           # seeds this unit's component tolerances (±2% tuning, ±5% decays, ±1 dB levels)
accent = 0.6                # the global ACCENT knob, 0..1
velocity = "continuous"     # or "accent": two trigger levels split at velocity 100
toms = "toms"               # or "congas" (808 panel switch for keys 41/43/45)
swing = 0.5                 # Linn/MPC swing of the second 16th in each pair; or "linn54" .. "linn71"
swing_unit = 16             # 8, 16 or 32
humanize_ms = 0             # Gaussian onset spread; 0 keeps the machine exact
knobs = { bd = { decay = 0.7, tune = 0.4 }, ch = { tone = 0.6 } }
# seed = 808, output_gain = 0.5, preset = "default"
```

Use one MIDI track per machine; any channel.
Every note needs an explicit note-off. Note length is ignored after the hit except for the 808's key 35 (the long boom, gated at note-off) and the bass track.
A key a machine lacks is a render error, never silence.

## Key map

| MIDI | LilyPond `\drummode` | 808 | 909 | Linn |
|---|---|---|---|---|
| 35 | `bda` | BD, long boom (T60 `bd.long_t60`, closed at note-off) | BD without the attack click | Kick, soft (a gentler stroke) |
| 36 | `bd` | BD | BD | Kick |
| 37 | `ss` | Rimshot | Rimshot | Side stick |
| 38 | `sn` | Snare | Snare | Snare |
| 39 | `hc` | Clap | Clap | Claps |
| 40 | `sne` | Snare, snappy +0.3 | Snare, tone +0.3 | Snare rimshot |
| 41 / 43 / 45 | `tomfl` / `tomfh` / `toml` | Low/mid/high tom (or conga) | Low/mid/high tom | Low/mid/high tom |
| 42 | `hh` | Closed hat | Closed hat | Closed hat |
| 44 | `hhp` | Pedal hat (closed, decay ×0.6) | Pedal hat | Pedal hat |
| 46 | `hho` | Open hat | Open hat | Open hat |
| 49 | `cymc` | Cymbal | Crash | Crash |
| 51 | `cymr` | — | Ride | Ride |
| 56 | `cb` | Cowbell | — | Cowbell |
| 70 | `mar` | Maracas | — | Cabasa |
| 75 | `cl` | Claves | — | — |

Keys 42 and 44 choke a ringing open hat in about 10 ms (the VCA discharging, not a click).

**`tr808-bass`** plays sounding pitches, MIDI 24–60 (C1–C4), monophonic, exactly in tune (the unit's tolerance does not detune it).
Each note re-excites the resonator and glides from the previous pitch over the `bass.glide_ms` before its onset, so a boom slides into each new root.
A note that starts before the previous one's note-off is legato: it glides at its onset without a new pulse (set `bass.retrigger = true` to re-excite anyway).
Note-off closes the boom over about 30 ms; write the note as long as it should ring.

## Velocity is a trigger level

On the 808 and 909, velocity is the trigger voltage, not the output gain.
Velocity 96 is a normal step (trigger 1.0); below it the trigger falls proportionally (softer and darker); above it the ACCENT knob adds up to `0.8·accent` at velocity 127.
A harder trigger deepens the bass drum's pitch blip, drives its clipper (more 2nd and 3rd harmonics) and makes the snare's noise bark.
Hits on the same step pull 15% toward that step's strongest trigger, and each extra accented voice on a step sags the accent bus by 4%: that is the box's accent coupling.
The Linn's velocity drives its analog VCA: gain = (velocity/100)^1.5. Its ROMs never change timbre with velocity, like the hardware.

Useful velocities: 96 for a normal step, 124–127 for an accent, 40–60 for a ghost (808 ghosts are a model extension; the real box has two levels).

## Knobs

Knobs are the panel's 0–10 divided by 10. Set them per piece in `knobs = { voice = { knob = value } }`; any circuit value in the preset can be overridden the same way.

| Voice | Knobs (0..1 unless noted) |
|---|---|
| `bd` (808) | `decay` (T60 0.15–1.6 s), `tune` (44–56 Hz), `tone` (click brightness), `click`, `drive`; also `decay_s` (exact T60, overrides decay), the slow sweep `sweep_semitones` (6.7) over `sweep_ms` (74), the fast attack chirp `snap_semitones` (24) with time constant `snap_ms` (3), `long_t60` (key 35) |
| `bass` (808) | `decay_s` (1.4), `sweep_semitones`, `sweep_ms`, `snap_semitones` (0), `click`, `tone`, `drive`, `glide_ms` (60), `retrigger` |
| `sd` (808) | `tone` (low body vs high body and noise brightness), `snappy`, `decay_s` (noise T60) |
| `cp` | `spread_ms` (burst spacing), `tail_s`, `tail`, `bursts` |
| `ch`, `oh` (808) | `decay` (CH T60 0.06–0.2 s, OH 0.25–1.6 s) or `decay_s` (exact T60), `tone`, `drive` (metal intermodulation) |
| `cy` (808) | `decay` (0.4–2.5 s), `tone` (low band vs high band), `top` (dark-able top) |
| `lt`, `mt`, `ht` | `tune` (±6 semitones), `decay`, or `hz_abs` for an exact pitch (e.g. toms tuned to D3/F♯3/A3) |
| `bd` (909) | `tune`, `decay`, `attack` (click), `sweep_ratio`, `sweep_ms`, `drive`, `shaper` |
| `ch`, `oh`, `cy`, `rd` (909, Linn) | `tune` (ROM playback clock: pitch and length move together), `decay` (the VCA after the DAC), `tone` |
| every voice | `level` (post-VCA, linear), `pan` (−1..1) |

**Knob rides** change a knob over a span of score bars (inclusive), later rides winning:

```toml
[tr808_model]
rides = [{ voice = "bass", knob = "drive", bars = [53, 68], value = 0.35 }]
```

**`tone` on hats, cymbals and maracas is the anti-harshness control.** At 1 the hat has the circuit's own spectrum: a TR-808 closed hat puts most of its A-weighted energy in 6–10 kHz. Lower tones close a 4-pole low-pass at `2 kHz · 10^tone` (11 kHz at the 808 default 0.75, 6.3 kHz at 0.5), which takes the fizz above 10 kHz first and leaves the 6–10 kHz body until the corner reaches it.
For a long listen, run 808 hats at `tone` 0.5–0.6 and at least 12 LU under the kick.

## Clock

A machine track is the room's fixed point: the performer places every hit on the score's tempo map exactly (rit and fermata included), with no humanize jitter, groove, room offset or expression, and passes velocities through unchanged.
Swing is composed: `swing = 0.58` delays the second 16th of every pair to 58% of the pair (Linn/MPC semantics), and positions between gridlines move proportionally so note-offs stay after their onsets.
Use the LM-1/LinnDrum names `"linn50"`, `"linn54"`, `"linn58"`, `"linn62"`, `"linn66"`, `"linn71"` for the historical settings; 54–62% is the useful range for 16ths.
`humanize_ms` is there for a deliberately sloppy machine; leave it at 0 for a machine that is a machine.
The machine does not yet act as the room's lead: in a piece with `[room]`, the humans follow the room's own lead, not the machine (open issue).

## Balance and limits

At default levels a classic 808 pattern sits around −25 LUFS unmastered, with the kick at the groove's loudness, snare and clap 6–8 LU under it, cowbell about 10 LU under and 8th-note hats about 16 LU under.
The 808 bass track is much louder than the kit (a 1.4 s sine every two beats); start it at `level = 0.4` or `gain_db = -8` beside a kit.
The renderer refuses clipping; lower `output_gain` if many accented voices coincide.

For the beat of *Machine Lullaby* on the album *Era*: `accent = 0.7`, `bass = { decay_s = 1.4, sweep_semitones = 12, sweep_ms = 35, click = 0.25, tone = 0.35, drive = 0.15 }`, `ch = { tone = 0.45, decay_s = 0.04 }`, rim and hat 12 and 24 LU under the boom.

Limits: no individual outputs yet (one stereo stem per machine), no `[machine]` step-pattern section in the sidecar (patterns come from LilyPond or MIDI), no flam or ratchet keys, rides step per bar rather than ramp, no `"line"`/`"speaker"` placement, no thermal drift, no Linn MPC quantizer or note repeat.
The 909 references available were imitations, so the 909 is built from the spec's circuit description more than fitted; the Linn approximates the LM-1 hits in spirit (its snare's wire band sits about 15 dB hotter in 1–3 kHz than the reference hits').
