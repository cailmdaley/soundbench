# Playing the physical guitar

Use sounding pitches, not octave-transposed notation pitches.
One render contains one shared guitar: all guitar-model tracks share six strings and one body.
Steel and nylon can't coexist in the same render; render them separately.

```toml
[instruments]
guitar = { model = "guitar", gain_db = -6, pan = -0.35 }

[guitar_model]
preset = "nylon"                 # steel strings by default
# output_gain = 20.0              # example absolute engine gain override, not dB
# tuning = "dadgad"               # standard, dadgad, open-g, or six MIDI opens
# capo = 2                        # 0..12; score still uses sounding pitches

[guitar_strings]
guitar = 1                       # optional track pin, 1 = highest string
```

Standard open strings, **1 through 6**: E4 B3 G3 D3 A2 E2 (MIDI **64 59 55 50 45 40**).
Each permits frets 0..20 above its tuned/capo open pitch.
DADGAD opens are 62 57 55 50 45 38; open G is 62 59 55 50 43 38.
A custom `tuning` array has the same high-to-low order.
Capo raises the open pitches; it does **not** transpose incoming MIDI.

Ordinary MIDI note-on plucks.
Open strings ring naturally through note-off; only retriggering or explicit damping stops them.
Fretted note-offs use register-dependent finger damping: primary T60 0.65/0.55/0.45/0.35 s at 80/200/500/1200 Hz, with a weaker polarization decaying three times more slowly.
T60 means 60 dB attenuation; the primary loses about 20 dB in 117–217 ms before adding the string's intrinsic losses.
**CC64 ≥64 holds MIDI notes; CC64 <64 submits their note-offs to this physical release policy.**
For track-level sustain, use the same option as the banjo:

```toml
[tracks]
guitar = { let_ring = true }
```

This uses the performance pass's harmony/rest-aware sustain: fretted notes begin finger damping at the extended note-off, including harmony changes.
`strum` and `rasgueado` honour each chord tone's extended note-off without adding the stroke offset to its release time.
Their release-policy flag does not create sustain windows; enable track `let_ring` to extend scored holds.
Open strings still require an explicit damp.
The renderer intentionally damps the instrument halfway through the configured final tail.
There are no keyswitches and no other interpreted CCs or pitch-wheel messages.
Track names have no magical suffixes: use selectors and string pins explicitly.
Selectors accept complete names, colon-separated voice/staff components, or one-based note-track ordinals.
Without pins the allocator prefers a free playable string with the lowest fret; if all candidates are occupied it steals the oldest.
Use separate named LilyPond Voices, each on its own MIDI track, to keep fingering unambiguous.

## Score-addressed gestures

Use `[[guitar_articulation]]`, **not** the banjo's `[[articulation]]`.
Bar and beat are one-based in the scored MIDI; the renderer maps the addressed onset to performed timing.
Only one gesture is allowed per track/onset.
Except for `damp`, a gesture must address an exact note/chord onset.
This table lists every supported kind and its extra fields.

| Kind | Request and behavior |
|---|---|
| `pluck` | One scored note; optional `string = 1..6`, `hardness = "preset" / "pick" / "flesh" / "nail"`, `position = 0.01..0.49` (fraction of speaking length). `pick` uses the preset; flesh is softer, nail harder. |
| `strum` | One scored chord on one track: a hand crossing the strings. `strings = [...]` pairs strings with chord pitches **highest first**; omit for allocation. `direction = "down" / "up"`. Without `speed_ms` the stroke takes the hand's time across the strings it spans: 30 ms for six strings at velocity 90 and 100 bpm, shorter when louder or faster (`presets/guitar/data/stroke.toml`), with seeded variation. Gaps lengthen after heavier strings and force falls across the stroke. `speed_ms = 10..80` instead fixes the first-to-last time. Upstrokes reach the top three or four strings; `reach = n` sets how many strings a stroke strikes from its starting side. `mute = T60` (0.005..2 s) damps the struck strings 18 ms after each sounds: a palm-muted strum (for a choked brush use `chuck`). `accent = 0..1.5`, strictly positive, scales the first string only (default 1). Optional hardness. Down goes from low to high strings. The earliest performed chord note supplies a common velocity. Fretted chord tones begin damping at their sustain-extended note-offs; open tones ring until retrigger/damp/end. |
| `chuck` | Same fields; each string is choked 18 ms after it sounds, with 35 ms T60, and released at 65 ms. A pitched muted brush, not a recorded percussive slap. |
| `rasgueado` | Same fields; `fingers = 1..4` (default 3). Sweeps begin at least 35 ms apart, with a 3 ms turnaround gap after slower strokes; they alternate direction and lose 8% force per finger. Start with `speed_ms = 18`, `hardness = "nail"`, velocities 65–85. |
| `palm-mute` | One note, optional string/hardness/position; damp begins 30 ms after attack with 35 ms T60. |
| `damp` | At an arbitrary score position after notes; no scored onset needed. Optional `string`; without it damps the latest preceding note on every string used by that track. Token checks protect notes subsequently stolen by another track. |
| `hammer` / `pull` | One scored destination note on an **explicit/pinned string**; previous note on that same track/string must remain held until the destination. Hammer goes up, pull goes down. Reuses the vibrating string with a softer excitation; source's stale note-off can't release the destination. |
| `slide` | One starting note; `to` is relative semitones, `duration_beats > 0`, optional `shape = "linear" / "ease"`. Target must fit the same string. Score the starting pitch only; the renderer moves it. |
| `bend` | Same as slide, **steel only**, `to = -1..1`. Small bends; not blues-guitar whole-tone bends. |

```toml
[[guitar_articulation]]
track = "guitar"
bar = 3
beat = 2
kind = "rasgueado"
strings = [2, 3, 4, 5]           # for chord C4 A3 E3 A2, highest pitch first
speed_ms = 18
direction = "down"
fingers = 3
hardness = "nail"
accent = 1.15
```

For generator-layer events, import `note`, `strum`, and `render` from `soundbench.scripts.models.guitar`.
`note(on_seconds, off_seconds, pitch, velocity, string, token, ..., kind=0, hardness=0, let_ring=False)` returns attack and MIDI-off rows.
`strum(..., let_ring=True)` honours the supplied release time for fretted notes; chucks explicitly damp.
Use `models.guitar_release.finish(events, at_seconds)` to add deliberate end-of-phrase damping to a standalone render.
On release rows, column 8 is 0 normal, 1 sustain-extended off, or 2 force finger release (including open strings); kind 5 is the stronger explicit choke, and its column 9 is a T60 in seconds (0 for the preset's choke).
`strum(on, duration_seconds, pitches, config, ..., strings=[...])` requires explicit string/pitch pairs; pairs may be in any order.
Tokens must be unique across calls; reserve **24 tokens per strum**.
`strum(..., tempo_bpm=100, reach=None, muted=False, humanize=True)`: `muted` is `True` for a chuck or a T60 for a palm-muted strum; `humanize=False` gives the hand's mean timing and force.
Kinds are 0 pluck, 1 release, 2 pitch ramp, 3 hammer, 4 pull, 5 damp; hardness codes are 0 preset, 1 flesh, 2 nail.

## Useful ranges and limits

- Velocities **60–105** for single notes; **65–85** for multi-finger strums; 110–127 for occasional accents.
- Steel boom-chuck: hold bass 0.4–0.7 s, brush the upper 3–4 strings with `chuck`, then release before the next bass.
- Nylon punteado: pin the melody to string 1; omit string 1 from accompaniment chords to preserve it.
- Fingerpicking: pin one voice per string; open notes ring naturally, and `let_ring = true` holds fretted ones until the harmony/rest boundary. End standalone renders with `finish`.
- Single-string slides can sound synthetic at delay-boundary crossings; use short, small moves.
- A pluck on a ringing string adds to its motion, as a pick on a real string does: the pick's 1.5 ms contact (`excitation.contact_ms`) leaves `restrike_keep` of the old wave. Before a pluck at a new pitch the fretting hand damps the old note `strings.refret_ms` earlier, with T60 `strings.refret_t60`. Damping glides in over `strings.damping_glide_ms`; nothing cuts off in a step. A pick, a hammer-on or a pull-off lifts any damp on its string at once, so a restrike is as loud as a fresh pluck; a slide carries the string's damping with it, so a slide between a refret and the next pluck cannot cancel the refret. Every damping T60 is at least 5 ms.
- Boom-chuck: keep the thumb's bass string out of the brush; a damp on that string would stop the bass too.
- Occasional short squeaks on slides of at least two semitones are enabled; `excitation.slide_noise` sets their gain (0 disables).
- Harmonics, body taps, barre squeaks and continuous palm-pressure control are **not implemented**.
- An impossible fingering fails rather than silently octave-shifting a note.
