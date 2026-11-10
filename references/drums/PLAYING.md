# Physical kit: playing sheet

Use one MIDI track for the whole kit, on any channel (channel 10 / zero-based 9 is conventional).
Don't split the hats across tracks: their shared pedal/choke lives within one track and channel.
The engine renders at 48 kHz, with fixed gain and a stereo close/overhead image.

```toml
[instruments]
kit = { model = "drums", gain_db = 0, pan = 0 }

[drums_model]
parameters = "instruments/models/drums.toml"
seed = 1974
# output_gain = 0.32
```

`drums.toml` is the short felt-muffled, ported funk kit.
`drums-open.toml` has an unported, unmuffled, softer-beater kick with a longer decay.
`drums-rock-fitted.toml` is the recording-fitted rock alternative; it isn't a validated funk or jazz kit.
Keep `pan = 0` to retain the internal image (hats left, ride/floor tom right, kick center).
The room stage applies once to the mix; the Nagra tape stage deliberately mono-sums the kit at its room microphone.

| MIDI key | Gesture | Useful velocity |
|---|---|---|
| 35, 36 | Kick (same drum) | 70–110 funk; 45–70 under brushes |
| 38 | Snare batter, center | 85–110 backbeat; 18–40 ghost notes |
| 39, 40 | Rimshot: batter + rim (39 repurposes GM hand clap) | 80–115 |
| 37 | Cross-stick: rim only | 55–95 |
| 41, 43, 45 | Low/floor tom (one drum) | 65–110 |
| 47, 48, 50 | High/rack tom (one drum) | 65–110 |
| 42 | Closed-hat stick hit; with CC4, hit at the current pedal position | 45–80 |
| 46 | Open-hat stick hit; with CC4, hit at the current pedal position | 45–90 |
| 63 | Half-open hat extension; with CC4, hit at its current position | 45–80 |
| 44 | Foot chick; closes and chokes the two plates | 40–85 |
| 51, 59 | Ride bow (one cymbal) | 50–90 |
| 53 | Ride bell | 65–100 |
| 49, 57 | Crash (one cymbal) | 70–105 |
| 60 | Brush sweep toward center; **hold the note for the entire sweep** | 40–80 |
| 61 | Brush return sweep near the edge; **hold the note** | 40–80 |
| 62 | Brush tap | 45–90 |

All other keys fail rather than substitute a sound.
This is a GM subset with explicit extensions, not a full GM percussion bank.
There are no pitch keyswitches, bends, pressure CCs or track-name articulation conventions.

## Pedal and release

**CC4: 0 = fully open, 64 = half-open, 127 = tightly closed.**
A CC4 value takes control on that MIDI channel until another CC4 event changes it.
It changes contact and damping of already-ringing hats, even after their MIDI note-offs.
Before the first CC4, keys 42/46/63 set the shared closed/open/half pedal position; key 44 always closes it.
To breathe the hats, play key 42 eighth notes, set CC4 to 90 on an offbeat, then return it to 127 about 100–180 ms later.
For a foot chick, write key 44 as well as any desired CC4 automation; CC4 alone doesn't generate a chick.
The last CC4 at a tick applies to every hat attack at that tick; emit it before the hit for portable MIDI playback.

**Every note needs an explicit note-off.**
Use 50–150 ms MIDI lengths for struck gestures: release doesn't choke their natural acoustic decay.
Only brush sweeps use held duration; release stops friction over 20 ms, leaving the head's natural tail.
A 90-bpm folk shuffle works with 1.0–1.2 s alternating sweeps and brush taps on beats 2 and 4 plus their last-triplet pickups.
Ghost-note brightness changes continuously with velocity; there's no sample-layer switch.

## LilyPond and generator layer

A LilyPond `DrumStaff = "kit"` drives the kit; in `\drummode`, use `bd`, `sn`, `sidestick`, `tomh`, `toml`, `hh`, `hho`, `hhp`, `ridecymbal`, `ridebell`, `crashcymbal`.
Built-in drum notation doesn't provide these brush extensions: a MIDI generator should emit keys 60–62 and CC4 explicitly.

## Balance and limits

The unmastered funk stem is about −26 LUFS at default gain, before the ensemble's shared headroom.
Brush touch is much quieter; bring up the **whole kit track** about 12–15 dB in a brush-only arrangement, after checking peaks.
Velocities 1–127 work, but simultaneous fortissimo drums can require a smaller `drums_model.output_gain`; the renderer refuses clipping.
The ride/crash are conspicuously modal, and the brush texture is uncalibrated friction noise rather than measured bristle mechanics.
Don't expect convincing full-GM playback, cymbal hand chokes, rolls from a held stick note, or continuous movement of a brush stroke across the head.
