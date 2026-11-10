# Voice2 — composer sheet

`voice2` is an alternative to `voice`, with a seeded singer and the same waveguide tract.
Write sounding pitches and attach lyrics with LilyPond `\lyricsto`.
The target is hums, vowels and cries, not intelligible words.
Use one monophonic line per track; simultaneous notes fail.
Overlaps are cut at the next onset and retain legato pitch motion.

```toml
[instruments]
voice = { model = "voice2", gain_db = 0, pan = 0 }

[voice2_model]
preset = "alto"
vibrato = 0.8
breathiness = 1.0
tension = 0.0
seed = 1961
release_s = 0.22
tail_seconds = 1.0
```

## Piece controls

Each setting is optional.
Unknown keys, nonfinite values and out-of-range values fail.
The parameter files are `soundbench/instruments/models/voice2.toml` and `voice2-alto.toml`.
They inherit the walls and MRI vowel shapes from `voice.toml` without modifying the `voice` model.

| Control | Range / default | Spectrogram cue | Sidecar example (inside `[voice2_model]`) |
| --- | --- | --- | --- |
| `preset` | `tenor` (default), `alto` | Alto formant envelope sits higher; passaggio also sits higher. | `preset = "alto"` |
| `vibrato` | 0–3 / 0.8; extent multiplier | Wider or narrower wandering harmonic rails; 0 leaves the small flutter, not a mathematically straight tone. | `vibrato = 0` |
| `breathiness` | 0–4 / 1; aspiration multiplier | Noise between harmonic rails, strongest in soft vowels. | `breathiness = 1.5` |
| `tension` | −1..1 / 0; subtracted from Rd | Positive values press the closure and brighten the comb; negative values soften it. Rd clamps to 0.3–2.7; hum Rd clamps to 1.15–1.4. | `tension = 0.3` |
| `seed` | nonnegative integer / 1961 | Changes vibrato delays, wandering, swells, dip widths and noise, not the written melody. | `seed = 20261010` |
| `release_s` | 0.01–5 s / 0.22 | Changes the fading rails after phrase ends; closure remains held through hum release. | `release_s = 0.5` |
| `tail_seconds` | 0.1–10 s / 1; must contain release | Render duration after the last note; too short a tail cuts the release. | `tail_seconds = 1.5` |
| `cry_lift_db` | −6..18 dB / 3 | How far an open-vowel cry rises above its hum. | `cry_lift_db = 6` |

Tenor uses 17.5 cm / 24 oral sections and a 330–520 Hz chest/head blend.
Alto uses 15.3 cm / 21 oral sections, a scaled nasal tube and a 440–700 Hz blend.
B3–B5 and velocities 30/80/127 form the stability grid; this is an engineering range, not a claim about a human tenor's usable compass.
There is no vowel-formant tuning toward high f0.

### Lyrics, dynamics and defaults

| Score control | Range | Spectrogram cue | Example |
| --- | --- | --- | --- |
| Lyrics `hm`, `mm` | held /m/ | Low nasal comb, live closed oral side branch, widened velum. | LilyPond `\lyricmode { hm }` with the binding above. |
| Lyrics `nn`, `ng` | held /n/, /ŋ/ | Different oral antiresonance from alveolar or velar closure. | `\lyricmode { ng }`; no extra sidecar key. |
| Lyrics `ah eh ee oh oo uh aw ae ai` | vowel or diphthong | Open formant envelope; `ai` slides from /ɑ/ toward /i/ late in its span. | `\lyricmode { ah }`; `cry` below can override its vowel. |
| No new lyric | melisma | Continues the previous syllable on the next scored pitch. | LilyPond lyric extender `__`; no sidecar key. |
| Velocity | MIDI 0–127; useful 40–100 | Changes level and source brightness together; loud long notes have brighter upper rails. | Write dynamics in the score, e.g. `\p` / `\f`; no sidecar velocity key. |
| CC11 | 0–127, default 127 | Interpolated effort/loudness arc; 0 mutes voicing but does not suppress an independent inhalation. | Send CC11 on the voice's note track; no sidecar CC11 key. |

A vowel `ah`, `ai` or `oh` anywhere in a hummed render is treated as a cry: glottal entry, minimum 0.85 effort, quick opening from /m/, then closure.
This also recognizes a window beginning at the cry rather than at its preceding hum.
Its level comes from the phrase's hum, not its own velocity: the cry sits `cry_lift_db` (default 3 dB, A-weighted) above the hum it opens from. The model sings each vowel against its hum once per preset to learn how much louder an open mouth radiates at each pitch (10–35 dB) and takes that off, so the cry is balanced at B5 as at D4. CC11 still shapes it with the rest of the line.
No `ah_gain_db` patch is read by voice2.
A standalone all-vowel line does not automatically become cries.
Known `voice` word syllables use their first vowel only; consonants and intelligibility are not supported targets.

Notes shorter than 0.5 s have no ordinary vibrato.
Long notes draw delays between 0.2 and 0.8 s, rate around 5–6 Hz with wandering, and a 2–6 dB swell.
Legato hums dip 2–8 dB for 40–90 ms at each new note, with deeper dips on repeated pitches.
Phrase gaps of at least 0.5 s receive an inhalation lasting 250–600 ms, subject to available room.
Phrase endings fall 20–80 cents and soften into release; B3 endings can enter creak.
All these gestures are chosen approximations, not fits to a particular singer.

## Score-addressed articulations

Every example is an independent `[[articulation]]` entry.
`bar` is one-based; `beat` defaults to 1 and uses the meter's beat unit (eighths in 6/8).
Every articulation addresses exactly one scored note onset, then follows that note's performed onset after rubato and groove.
`duration_beats` interpolates score/performed anchors, as for the banjo.
Different kinds can coexist at one onset; duplicate kinds fail.
Unknown kinds and keys fail, including inappropriate keys on a known kind.
For a custom track name, replace `track = "voice"` in the examples.

### Slide

`to` is −12..12 semitones relative to the scored pitch; `duration_beats` must be positive and map to 0.005–10 s.
The destination holds until the next note.
The spectrogram shows a continuous eased bend of every harmonic.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "slide"
to = 2
duration_beats = 0.5
```

### Cry

`vowel` is one of the vowel lyric names above, default `ah`.
The spectrogram shows the nasal comb opening into a bright vowel, then returning to closure, with a hard voicing entry.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "cry"
vowel = "ai"
```

### Onset

`style` is `breathy`, `glottal`, `scoop` or `hum`.
`depth` is 0–8 semitones, used by scoop; default scoop is drawn 0.5–1.5 semitones, or 0.3 for hum.
Breathy shows 55 ms of noise before the rails, glottal uses a 10-ms hard rise, scoop bends up over 160 ms, and hum uses a 55-ms rise without the aspiration lead.
`hum` chooses the onset envelope, not the sustained syllable: write `hm` to keep the lips closed.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "onset"
style = "scoop"
depth = 6
```

### Fall

`depth` is 0–6 semitones, default drawn 0.2–0.8.
Optional `duration_beats` maps to 0.005–10 s, default 0.2 s.
All harmonic rails bend downward at the end of the addressed note while level falls 4 dB and Rd rises.
It can be placed on a nonfinal note.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "fall"
depth = 4
duration_beats = 0.5
```

### Creak

No extra keys.
The final 120–160 ms relax toward irregular 30–70 Hz pulse spacing with alternating strength.
The spectrogram shows widely spaced pulse transients and subharmonic energy, rather than ordinary vibrato.
High-note creak is permitted for experimentation but not physically validated.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "creak"
```

### Breath

No extra keys.
A 320-ms inhalation ends 30 ms before the addressed onset; the beginning truncates at time zero.
The spectrogram shows broad noise in the preceding gap.
Place it after a real rest: forcing it before a connected note also opens the tract during the previous note.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "breath"
```

### Straight

No extra keys.
Disables ordinary vibrato only on the addressed note; flutter remains.
The spectrogram shows almost straight rails with a few cents of slow drift.
An explicit quiver still works on the same note.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "straight"
```

### Quiver

`depth` is 0–6 semitones of half-extent, default 0.6; about 0.5–1 is the intended idiom.
Rate is drawn 5–7 Hz with a 100-ms rise.
The spectrogram shows a much wider ornament than ordinary vibrato.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "quiver"
depth = 0.6
```

### Flick

`to` is −12..12 semitones, default +2.
The pitch makes a brief excursion centred 100 ms into the note with a 25-ms width.
The spectrogram shows a compact hook on the rails before the sustained pitch resumes.

```toml
[[articulation]]
track = "voice"
bar = 5
beat = 1
kind = "flick"
to = 2
```

## Reproduction

The spectrogram cues above are visual checks, not proof of perceived realism.

```sh
uv run python -m soundbench.scripts.voice_phrases --model voice2 --preset tenor --out <dir>
uv run python -m soundbench.scripts.voice_phrases --model voice2 --preset alto --out <dir>
uv run tests/test_voice_model.py
uv run tests/test_voice2_model.py
```
