# Tonewheel organ: playing sheet

Use `model = "organ"`; MIDI **36–96 (C2–C7)**, including chromatic notes.
There are no keyswitches, program-change presets or track-name gestures.
Give every note a note-off; CC64 does **nothing** on this organ.
Velocity 1–127 starts a key but does not change its loudness or tone.
Use CC11 for swells.
Each MIDI channel is an independent manual/percussion circuit; tracks render as separate preamps/cabinets, not one shared organ.
Put connected organ parts on one track/channel when their percussion should share single-trigger state.

```toml
[instruments]
keys = { model = "organ", gain_db = 0, pan = 0 }

[organ_model]
preset = "gospel"

[[controls]]
track = "keys"
bar = 5
beat = 1
control = 68
value = 127
```

`[[controls]]` uses one-based bar/beat positions in the score; the performance pass moves these onto performed time.
A MIDI generator can instead emit ordinary channel-matched `control_change` messages at absolute ticks.
Set switch controls before the relevant note-on, preferably in a rest.

| Gesture | MIDI request | Behavior |
| --- | --- | --- |
| Swell pedal | CC11 0–127; default 127 | Continuous volume, 10 ms smoothing; 0 silences the manual. |
| Leslie slow / fast | CC68 0 / 127; default 0 | Horn 0.8→6.7 Hz, drum 0.7→5.8 Hz; intermediate values set intermediate targets. Horn/drum e-fold acceleration times 0.8/2.4 s, same on deceleration. A switch is **not** an instant speed change. |
| Drawbars | CC12–20, in the order below | Set drawbar 0–8 with `round(127*digit/8)`: **0,16,32,48,64,79,95,111,127**. Updates held notes with 4 ms smoothing. |
| Scanner off, V1/V2/V3, C1/C2/C3 | CC1 **0,21,42,64,85,106,127** | Dispersive line taps scanned at 6.8 Hz; V is wet-only, C mixes wet/dry. This consumes CC1; it isn't pitch-bend. |
| Percussion on / off | CC80 ≥64 / <64 | Default off; enabling percussion disables the ninth/1-foot drawbar. |
| Percussion second / third | CC81 <64 / ≥64 | Octave / octave-plus-fifth gear-tuned wheel; set before the phrase. |
| Percussion slow / fast | CC82 <64 / ≥64 | Amplitude e-fold decay 0.65 / 0.16 s. |
| Percussion normal / soft | CC83 <64 / ≥64 | Soft is 0.55 of normal percussion amplitude. |
| Click / overdrive | Note on/off; preset TOML | Microcontact bounce and release click are automatic. `contacts.click_gain` and `preamp.drive` are configuration controls, not MIDI gestures. |

Drawbar order: **16′, 5⅓′, 8′, 4′, 2⅔′, 2′, 1⅗′, 1⅓′, 1′**.
Their offsets from the scored key are **−12,+7,0,+12,+19,+24,+28,+31,+36** semitones.
The bank uses actual 60-Hz gear ratios and octave tooth doublings, not equal temperament.
Harmonics above F♯8 fold down by octaves; the 16-foot contact on the bottom C2 uses C1.

## Registrations

| Preset | Registration | Use |
| --- | --- | --- |
| `default` | `888000000` | Full low drawbars; pop/soul comping. |
| `hollow` | `808000000` | Less fifth, hollow accompaniment. |
| `gospel` | `888800000` | Gospel comping, extra octave; louder than the default at the same gain. |
| `jazz` | `800000000` | Sub-octave drawbar plus soft, fast third-harmonic percussion. |
| `clean` | `888800000` | Gospel registration with lighter preamp drive. |
| `flute` | `006080300` | Fitted to a line-out flute recording, rotary off; not a B-3 flute claim. |

Percussion has one decaying envelope shared by all keys on that channel.
A chord struck from silence triggers once; a legato note joins the existing decay without restarting it.
Release **all** keys before the next attack to retrigger.
Each key's percussion component stops at that key's own note-off.
The implementation rearms immediately after all releases; it does not simulate a separate capacitor recharge time.

## Useful writing

Keep the bass in C2–C3, right-hand thirds/sixths/sevenths in C4–C6, and leave rhythmic air between chord stabs.
For gospel, switch the Leslie to fast at a phrase boundary and let the drum take several seconds to catch up.
For jazz percussion, use detached starts followed by legato fragments; a continuously held bass on the **same channel** prevents retriggering.
Organ dynamics written as velocity won't make a crescendo: write CC11 instead.
Changing all drawbars/scanner modes mid-note can expose switching transients; write these changes in rests unless that edge is intentional.
Extreme sustained full registrations and dense multi-channel chords can still clip; the renderer refuses clipping rather than silently limiting it.
