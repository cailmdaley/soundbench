# Playing the machine voice

The machine voice sings with the tract voice (`model = "voice"`), corrects its control-rate pitch, optionally speaks through another carrier, and adds band-limiting and distortion.
It uses no recordings unless a Python caller explicitly supplies an audio carrier.
A plain `model = "voice"` binding ignores the machine-voice settings.

## Binding and lyrics

```toml
[instruments]
voice = { model = "machine-voice", gain_db = 0, pan = 0 }

[machine_voice]
preset = "corrected"
sob = 0.7
# scale = [4, 6, 8, 9, 11, 1, 3]  # E major, MIDI pitch classes
# mix = 0.8

[voice_model]
release_s = 0.2
# tail_seconds = 1.0
# output_lowpass_hz = 10000.0
```

Score a monophonic voice and attach a LilyPond `\new Lyrics \lyricsto` context.
The renderer uses the plain voice's lyric-to-note matching and pronunciation lexicon.
`hm` keeps the lips closed and the nasal branch open; `ah` is an open /ɑ/ vowel.
A note without a lyric continues the previous syllable as a melisma.
The first note needs a syllable.
Track selectors accept a complete MIDI track name, colon-separated components, or a one-based note-track ordinal.

The `[voice_model]` settings retain the plain voice renderer's validation and meanings, including cut-only `ah_gain_db` registration.
They are separate from the processing settings below.
The voice parameters, effective machine parameters, explicit gestures and piano carrier data are included in hashed metrics provenance.
The stem metrics contain the effective scale, including a scale taken from scored MIDI's first key signature.
With no key signature or explicit scale, the fallback is E major.
A minor key signature selects its natural minor scale.

## Processing parameters

Every key accepted in `[machine_voice]` is listed here.
Unknown keys are errors, including keys in a parameter file's preset tables.
Numeric values must be finite; `bands` and `seed` must be integers.
Defaults below are the base defaults; the selected preset modifies them as listed in the next section.

| Key | Range / choices | Base default | Meaning |
|---|---|---|---|
| `preset` | `corrected`, `vocoder`, `talkbox`, `speaking-piano` | `corrected` | Select base sound before overrides. |
| `scale` | Nonempty list of distinct integer pitch classes 0–11; C=0 | Piece key, fallback `[4,6,8,9,11,1,3]` | Scale grid in every octave; array order doesn't matter. |
| `strength` | 0–1 | 0.75 | Amount of correction; held ±40-cent vibrato leaves ±10 cents at 0.75. |
| `retune_ms` | 0–1000 ms | 40 | Time constant for the target glide; 0 is instant target retuning. Leaps wider than a whole tone retune faster, by 2/\|leap in semitones\|. |
| `hysteresis_cents` | 0–100 cents | 15 | Pitch must pass the midpoint by this margin to change target. |
| `glitch` | 0–1 | 0 | Old-note-directed spike at each accepted target change: 20–40 ms and 40–60 cents at 1. |
| `mode` | `none`, `channel`, `lpc` | `none` | Corrected voice, channel vocoder, or LPC envelope cross-synthesis. |
| `carrier` | `voice`, `buzz`, `piano`, `organ`, `audio` | `voice` | Excitation for cross-synthesis; `none` always passes the corrected voice. |
| `bands` | Integer 4–64 | 20 | Log-spaced channel-vocoder bands. |
| `low_hz` | 20–1000 Hz | 100 | Lowest channel edge. |
| `high_hz` | 1000–20000 Hz, greater than `low_hz` | 8000 | Highest channel edge. |
| `attack_ms` | 0.1–1000 ms | 5 | Channel envelope attack time constant. |
| `release_ms` | 0.1–2000 ms | 40 | Channel envelope release time constant. |
| `formant_shift` | −24–24 semitones | 0 | Move the modulator envelope in frequency without moving carrier pitch. Applies in `channel` and `lpc`. |
| `sibilance` | 0–1 | 0.3 | Channel mode's top-band noise on high/low-ratio unvoiced frames. |
| `buzz_shape` | 0–1 | 0.25 | Saw (0) to narrow pulse (1), built from band-limited harmonics. |
| `buzz_detune_cents` | 0–100 cents | 3 | Second buzz oscillator's upward detune. |
| `lowpass_hz` | 100–20000 Hz | 3500 | Eighth-order output low-pass, after distortion and breakup. |
| `highpass_hz` | 0–2000 Hz, below `lowpass_hz` | 120 | Fourth-order output high-pass; 0 disables it. |
| `drive` | 0–1 | 0 | Four-times-oversampled asymmetric tanh saturation; RMS makeup before output filters. |
| `breakup` | 0–1 | 0 | Per-band thinning, 20–80 ms decimation bursts and target-change pitch spikes. |
| `mix` | 0–1 | 1 | Wet/dry linear mix; dry includes the performed sob, but not correction or colour. |
| `seed` | Integer 0–4294967295 | 1729 | Processing and waver randomness; the voice and piano retain their own configured seeds. |
| `sob` | 0–1 | 0 | Automatic phrase gestures, scaled by this amount. |

Increasing `drive` blends in progressively stronger tanh saturation, adding harmonics *inside* the requested output bandwidth.
It doesn't bypass the final low-pass.
`breakup` raises the effective pitch-glitch amount to at least its own value; `glitch` isn't multiplied by `strength`.
With both faults at zero, `strength = 0` leaves the input f0 unchanged.
`mix = 0` returns the performed dry voice exactly, including gesture headroom and any input controllers.
A dry contribution can restore high-frequency energy, so the bandwidth targets apply at `mix = 1`.

## Presets

All four use strength 0.75, retune 40 ms, hysteresis 15 cents, full wet mix, no automatic sob, and no glitches or breakup.

| Preset | Parameters changed from base defaults | Reason |
|---|---|---|
| `corrected` | `lowpass_hz = 3200`, `drive = 0.12` | Correction with the original voice carrier, a dark top and light roughening; its settings follow the measured chain of a reference recording. |
| `vocoder` | `mode = "channel"`, `carrier = "buzz"` | Twenty 4th-order band-passes give a recognizable channel-envelope experiment without an extremely coarse four-band vowel. |
| `talkbox` | `mode = "lpc"`, `carrier = "buzz"` | An order-24, 25 ms all-pole envelope speaks through a continuous harmonic carrier. |
| `speaking-piano` | `mode = "channel"`, `carrier = "piano"` | The modelled piano plays scale-snapped voice notes; its stereo channels are summed to mono before modulation. |

Eight versus 32 channel bands is a meaningful change, not an output-quality switch.
Fewer bands blend broader parts of the vowel spectrum; more bands preserve narrower spectral detail.
The piano carrier follows snapped scored notes, not the corrected continuous f0 or sob scoops.
Its strings decay while a long hum continues.

## Sob performer

The automatic rule groups contiguous notes into phrases using the voice configuration's `legato_gap_s` (default 0.12 s).
It selects the highest note of each phrase, breaking pitch ties by longest duration.
That note gets a cry and an end-of-note break.
Notes longer than 1.5 s get a mid-note catch, notes longer than 1 s a heave; subsequent phrases get a breath intake.
`sob` scales pitch offsets, level changes and breath strength.
Explicit gestures remain full strength unless their optional `scale` field says otherwise.

Use scored, one-based `bar` and `beat`, not performed seconds.
A gesture must select exactly one machine-voice track and an exact scored note onset.
The renderer matches that scored note by pitch order to its performed onset, so rubato and timing humanization carry the gesture with the note.
Only one explicit gesture is allowed per track/onset.
It replaces **all** automatic gestures addressed to that onset, including the automatic break.

Every gesture has `track`, `bar`, `beat` (default 1), `kind`, and optional `scale = 0..1` (default 1).
Unknown or irrelevant fields are errors.

| Kind | Fields: default; range | Expansion |
|---|---|---|
| `cry` | `from_cents = -100`; −1200–0. `rise_ms = 150`; 1–2000 ms. `swell_db = 9`; 0–24 dB. | On the note: rise from below with quadratic ease-out, swell over the same interval, hold until note-off. |
| `break` | `interval = -5`; −24–0 semitones. `fall_ms = 40`; 1–1000 ms. `drop_db = -16`; −40–0 dB. | Address the high note; execute at its **end**, fall relative to its pitch and hold the low landing until a later note onset. Level is relative to a cry at the same onset if one exists, otherwise relative to baseline. |
| `catch` | `depth_cents = -250`; −1200–0. `ms = 150`; 1–2000 ms. | Begin at the note's midpoint: sine-squared pitch dip and recovery over `ms`, with a simultaneous 6 dB level dip. |
| `breath` | `ms = 90`; 1–1000 ms. | CC2 intake immediately before the addressed onset, cutting phonation and adding whisper aspiration. At the start of the render it is truncated at time zero. |
| `waver` | `amount = 1`; 0–1. | Extra ±35-cent pitch and ±2 dB level modulation, irregular smooth 5–7 Hz, tapered over 50 ms at each end. Adds to ordinary voice vibrato; correction subsequently reduces it. |
| `heave` | `amount = 1`; 0–1. `rate_hz = 2.2`; 0.5–6. `dips_per_s = 0.8`; 0–3. `dip_db = -12`; −24–0. | Over the note: an irregular level heave (3.2 dB SD at full amount, smoothed at `rate_hz`) plus Poisson-placed 80–150 ms dips of `dip_db`, each with a −30-cent sag and a CC2 breath of up to 60. Drive compresses it: at drive 0.6 the output level measures 1.7–2.6 dB SD, against 2.3–3.1 dB in the reference recording. |

```toml
[[voice_gesture]]
track = "voice"
bar = 3
beat = 2
kind = "cry"
from_cents = -100
rise_ms = 150
swell_db = 9

[[voice_gesture]]
track = "voice"
bar = 4
beat = 1
kind = "break"                 # fall at this note's end
interval = -5
fall_ms = 40
drop_db = -16

[[voice_gesture]]
track = "voice"
bar = 5
beat = 1
kind = "catch"
depth_cents = -250
ms = 150

[[voice_gesture]]
track = "voice"
bar = 6
beat = 1
kind = "breath"
ms = 90

[[voice_gesture]]
track = "voice"
bar = 7
beat = 1
kind = "waver"
amount = 0.6
```

CC11 is bounded at 127, so the performer reserves the largest requested swell as track-wide headroom.
For a +9 dB cry, ordinary notes run at −9 dB expression and the cry rises to 0 dB.
This preserves the relative swell without sending invalid controller values or clipping CC11.
The gesture expression multiplies any existing CC11; it never replaces the original track's dynamics.
Pitch offsets add to any supplied `pitch_cents` lane; intake takes the maximum of gesture breath and existing CC2.
Gesture curves are expanded at 1 ms and then linearly interpolated by the voice.

## Python interface and voice contract

```python
from soundbench.scripts.models.machine_voice import load_config, render

config = load_config(preset="vocoder", overrides={"bands": 32, "sob": 0.5})
audio, controls = render(notes, seconds=5.0, config=config,
                         return_controls=True,
                         gestures=[{"kind": "cry", "time": 2.0}])
# For carrier="audio", pass carrier_audio=<finite mono array of seconds*48000 samples>.
```

Notes are the voice's dictionaries: `time`, `off` (caller seconds), `pitch` (MIDI, fractional allowed), `velocity`, `syllable` (or `None` for melisma).
Generator gestures use `time` instead of track/bar/beat; the Python layer permits multiple kinds at the same onset, such as a cry and a break.
`render` returns mono float audio at 48 kHz.
Machine controls include `t`, `input_f0`, corrected `f0`, `amplitude`, expanded `midi_controls`, and the placed `gestures`.
`amplitude` is the voice's pre-CC11 control track; the separate controller lane carries expression.
Machine control times exclude the voice's pre-roll.

The underlying voice interface is:

- `voice.render(notes, seconds, config, return_controls=True, midi_controls=..., f0_map=...)` returns audio plus controls including `t`, `f0` and `amplitude`.
- `f0_map(t, f0)` receives caller seconds (negative during engine pre-roll) and control-rate Hz; it must return an equally shaped, positive, finite Hz array.
  It is called before sample-rate interpolation and the glottal source.
- `pitch_cents` rows are linearly interpolated offsets applied before vibrato.
- CC2 rows accept 0–127: 0 leaves the voice unchanged, 127 shuts phonation off and puts aspiration at the whisper level.
- CC11 remains linearly interpolated source expression, 0–127.
- The voice's own returned `t` retains engine time, including pre-roll; only the mapping callback and input controller times use caller time.
  The machine layer translates that control timeline when it returns its own tracks.

Absent these hooks, the voice renders bit-identically to a render without them, including renders with existing CC11.
The engine tests compare float bytes, not a rounded WAV or a tolerance.

## Limits

- Cry and break replace the voice's ordinary pitch glide over their active span, retaining vibrato.
  The performer obtains the unmodulated baseline through a reference voice render with vibrato depth zero; it does not inspect source or tract internals.
  Correction acts **after** the gesture lanes: a nominal 40 ms sob fall of a fourth lands in about 58 ms (10–90%) after retuning, and a cry's ease-out becomes a scale-snapped trajectory.
  The lane timings are not a promise about the processed audio's timing.
- Headroom is track-wide; adding a cry reduces the baseline level of the whole track.
  Use the binding's gain to place the stem in the mix rather than pushing CC11 beyond 127.
- A break after a note ending a phrase can have little audible low hold because the voice releases at note-off.
  Score a continuing low note if you want the collapse to ring.
- `organ` fails with “organ carrier is not yet available”.
  `audio` is a Python array experiment, not a sidecar file loader.
- LPC is a stabilized all-pole envelope: diagonal loading and bandwidth expansion limit narrow resonances.
  Frequency-domain envelope filtering with Hann overlap-add and frequency warping approximates a talkbox; it is not a physical tube model and cannot reproduce nasal spectral zeros exactly.
- Channel noise uses a high/low spectral-energy ratio, not a speech classifier.
  It cannot recover missing consonants from the voice's limited pronunciation lexicon.
- Cross-synthesis uses global RMS makeup, not a compressor or limiter.
  Rendered peaks can differ between modes and should be checked before mastering.
- The measured reference distortion is entangled with stem-separation residue.
  Light saturation is a design choice, not a verified reconstruction of that recording's processor.
- The tests use synthesized f0, not pitch-estimator behavior on live microphones.
  The processing layer does not accept microphone audio as its modulator.
