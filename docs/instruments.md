# Instruments

Every model here synthesizes its sound; none plays samples.
A parameter file is a preset: one setting of a model, set by hand or fitted to some recordings. The defaults below are the presets Era uses; [the fitting guide](fitting.md) shows how to fit your own.
“Listenable on *Era*” means the instrument plays on the album *Era* with the settings named, and its listener found it listenable there.
“Approved” records a listening approval of a sound, not of every pitch, preset or articulation.
Both are human judgements; neither makes a model sample-exact or authentic to a particular physical instrument.

Run a row's probe as `uv run soundbench play MODEL --pitch PITCH --out build/probe.wav`.
The probe repeats one note at three velocities and includes its decay.
The playing sheet under `references/` documents each model's controls and playable range.

To play an instrument with its Era settings, put them in a TOML file under the `<model>_model` table and pass it with `--settings`:

```toml
# banjo.toml
[banjo_model]
parameters = "instruments/models/banjo-gut.toml"
```

The guitar takes `preset = "nylon"` (or `"steel"`) under `[guitar_model]`; the piano takes `parameters = "instruments/models/piano-close.toml"` under `[piano_model]`, or no settings for its default.

```sh
uv run soundbench play banjo --settings banjo.toml --out build/banjo-era.wav
```

A relative `parameters` path is looked up in the working directory first, then among the files bundled in `soundbench/instruments/models/`.

| Model | Probe | Settings on *Era* | Status and limits |
|---|---:|---|---|
| `piano` | 60 | default `piano.toml`; `piano-close.toml` on the darker tracks | Approved modal piano, listenable on Era. The default's modal, coupled-string and body statistics are fitted to Salamander grand-piano recordings; the model plays none of their samples. |
| `banjo` | 60 | `banjo-gut.toml` | Gut sound approved, listenable on Era. Steel and recording-chain variants are experimental. |
| `guitar` | 60 | preset `nylon`, or `steel` | Listenable on Era in both presets. The strummed tone has named flaws: it can sound dull beside a recorded guitar. |
| `fiddle` | 69 | `tuning = "ADAE"` | Listenable on Era. Some exposed tones are shrill and the bowing is approximate. |
| `mandolin` | 67 | default | Listenable on Era. Paired courses beat against each other; tone is provisional. |
| `pipe-organ` | 60 | `registration = "flutes"` | Listenable on Era with the flute registration; other registrations unreviewed. |
| `flabiol` | 84 | `flabiol-d.toml` or `flabiol-g.toml` (just scales on D and G) | Listenable on Era. Tuning is constructed and the jet is calibrated on a tin whistle, not a Mallorcan flabiol. |
| `xeremia` | 60 | `xeremia-d.toml` or `xeremia-g.toml` (D4 and G4 chanters) | Listenable on Era. Calibrated on Galician gaita recordings with constructed tuning; rests don't silence the chanter. |
| `skinpipe` | 60 | `skinpipe.toml` | Invented: a reed pipe whose bell is a drum skin, with sympathetic string courses. Approved, listenable on Era. Pressure bends pitch away from the courses. |
| `tambori` | 36 | default | Sound approved, listenable on Era. An unfitted physical tabor approximation. |
| `castanyoles` | 60 | `castanyoles-fitted.toml` | Sound approved, listenable on Era. Calibrated on orchestral castanets, not Mallorcan ones. |
| `ximbomba` | 48 | `ximbomba-fitted.toml` | Listenable on Era. A friction-drum class-proxy fit; local authenticity isn't established. |
| `voice` | 60 | default, wordless (`ah`, `hm`) | Basic tract sound approved for wordless singing, listenable on Era. Limited diction, synthetic tone. |
| `voice3` | 60 | singers `tenor`, `alto`, `bass` | Listenable on Era with words. Diction, an upper-register whistle and monotony of tone remain named limitations. |
| `chimera` | 60 | singers `bass`, `baritone`, `tenor`, `alto`, `soprano`; reads the `[voice3]` parts, characters and marks | Experimental: an articulatory singer (branched airway, aerodynamic glottis) from the CHIMERA fork in `vendor/chimera`; build it with Cargo first. Not yet heard on Era. Bodies are provisional, not fitted singers. |
| `machine-voice` | 60 | preset `vocoder`, 20 bands | Listenable on Era. Words depend on the mix around it. |
| `tr808`, `tr808-bass` | 36 | per-voice knob overrides (see the drum-machine sheet) | Listenable on Era. A circuit-inspired model with fitted hit parameters. |
| `voice2` | 60 | not on Era | Experimental vowels and cries; exposed high tones can be harsh. |
| `dobro` | 62 | not on Era | Sound approved. Square-neck construction and steel-bar noise are unverified. |
| `bass` | 43 | not on Era | Gut and steel upright presets; no listening approval recorded. |
| `electric-bass` | 43 | not on Era | Flat and round physical presets; no listening approval recorded. |
| `drums` | 36 | not on Era | Modelled kit with approximate brushes and cymbals; no listening approval recorded. |
| `organ` | 60 | not on Era | Drawbar registrations fitted to family proxies, not a verified tonewheel organ. |
| `electric-piano` | 55 | not on Era | Tine model fitted to family proxies, not a verified Rhodes. |
| `tr909` | 36 | not on Era | Built from the circuit description more than fitted; no listening approval recorded. |
| `linn` | 36 | not on Era | Procedural approximation of the LM-1 hits, not a ROM or sample set. |

The room, microphone and tape stage (`soundbench.scripts.models.nagra`) is approved and plays on Era with its `stage` and `basement` presets.
Its geometry, saturation and noise include unmeasured priors, and it ships no recorded impulse response.

Some Era parts are played by sampled instruments that are not in this library: the upright bass on most tracks, and the guitar and grand piano on a few.
