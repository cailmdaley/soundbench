# Instruments

Each instrument is one physical model with a `default` preset and sometimes named ones; a preset is a setting of the model, set by hand or fitted to recordings.
[The fitting guide](fitting.md) shows how to fit a preset of your own.

```sh
uv run soundbench presets banjo                                   # default, membrane, steel
uv run soundbench play banjo --preset steel --out build/steel.wav
uv run soundbench play banjo --preset my-banjo.toml --out build/mine.wav
```

From Python, `render("banjo", notes, preset="steel")`; in a settings file, `preset = "steel"` under `[banjo_model]`.
Presets live in `soundbench/presets/<instrument>/`.
A preset of your own can begin with `inherits = "banjo/steel"` and hold only what it changes.

Run an instrument's probe as `uv run soundbench play MODEL --pitch PITCH --out build/probe.wav`: one note at three velocities with its decay.
The playing sheet under `references/` documents each instrument's controls and playable range.
“Approved” records a listening approval of a sound, not of every pitch, preset or articulation; “on *Era*” means it plays on the album *Era* in the preset named and was found listenable there.

| Instrument | Probe | Presets | Status and limits |
|---|---:|---|---|
| `piano` | 60 | `default`: modal grand fitted to Salamander recordings; `close`: close-recorded and darker | Approved; both on Era. Plays none of the recordings' samples. |
| `banjo` | 60 | `default`: gut strings; `steel`: steel strings; `membrane`: the hand-set design with a modal head, where fits start | Gut approved and on Era; steel and membrane unreviewed. |
| `guitar` | 60 | `default`: steel strings; `nylon`: nylon strings | Both on Era. The strummed tone can sound dull beside a recorded guitar. |
| `bass` | 43 | `default`: gut upright; `steel`: steel upright | No listening approval recorded. |
| `electric-bass` | 43 | `default`: flatwounds; `round`: roundwounds; `warm`: dark, soft-fingered | No listening approval recorded. |
| `fiddle` | 69 | `default`: old-time fiddle; `modern`: brighter, untilted body | On Era with `tuning = "ADAE"`. Some exposed tones are shrill; the bowing is approximate. |
| `mandolin` | 67 | `default` | On Era. Paired courses beat against each other; tone is provisional. |
| `dobro` | 62 | `default` | Approved. Square-neck construction and steel-bar noise are unverified. |
| `organ` | 60 | `default`, `clean`, `gospel`, `hollow`, `jazz`: drawbar registrations; `flute`: flute drawbars with no leakage, key click or Leslie | Fitted to family proxies, not a verified tonewheel organ. |
| `pipe-organ` | 60 | `default`, with stops chosen by `registration` | On Era with `registration = "flutes"`; others unreviewed. |
| `electric-piano` | 55 | `default`: tine piano; `bark`: pickup closer to the tine, a harder attack; `tremolo`: with amplitude tremolo | Fitted to family proxies, not a verified Rhodes. |
| `flabiol` | 84 | `default`; `d`, `g`: just scales on D and G | On Era. Tuning is constructed; the jet is calibrated on a tin whistle, not a flabiol. |
| `xeremia` | 60 | `default`; `d`, `g`: D4 and G4 chanters | On Era. Calibrated on Galician gaita recordings with constructed tuning; rests don't silence the chanter. |
| `skinpipe` | 60 | `default` | Invented: a reed pipe whose bell is a drum skin, with sympathetic string courses. Approved, on Era. |
| `tambori` | 36 | `default` | Approved, on Era. An unfitted physical tabor. |
| `castanyoles` | 60 | `default`: fitted to orchestral castanets | Approved, on Era. Mallorcan construction is not established. |
| `ximbomba` | 48 | `default`: fitted to a Mallorcan ximbomba; `austria`: an Austrian friction drum whose hand lifts between strokes | On Era (`default`). Local authenticity isn't established. |
| `drums` | 36 | `default`: modelled kit; `open`: an undamped kick; `rock`: fitted to a recorded rock kit | Brushes and cymbals approximate; no listening approval recorded. |
| `tr808`, `tr808-bass` | 36 | `default`, with per-voice knobs (see the drum-machine sheet) | On Era. Circuit-inspired with fitted hit parameters. |
| `tr909` | 36 | `default` | Built from the circuit description more than fitted; no listening approval recorded. |
| `linn` | 36 | `default` | A procedural approximation of the LM-1 hits, not a ROM or sample set. |
| `chimera` | 60 | singers `bass`, `baritone`, `tenor`, `alto`, `soprano` and their characters, in a `[singing]` block | An articulatory singer (branched airway, aerodynamic glottis) from the CHIMERA fork in `vendor/chimera`; build it with Cargo first. Bodies are provisional, not fitted singers. |

The room, microphone and tape stage (`soundbench.scripts.models.nagra`, settings in `soundbench/rooms/nagra.toml`) is approved and plays on Era with its `stage` and `basement` presets.
Its geometry, saturation and noise include unmeasured priors, and it ships no recorded impulse response.
