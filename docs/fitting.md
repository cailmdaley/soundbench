# Fit an instrument to recordings

The banjo fitter shares physical parameters across isolated notes and can factor out a microphone/room/background/codec chain.
It uses a differentiable model during optimization and exports parameters for the runtime instrument.
Identifiability is limited: an excitation tilt, microphone colour and room decay can explain overlapping features.
A lower recording-match loss doesn't by itself establish a better dry instrument.

## Inputs

Install the fitting extra:

```sh
uv sync --extra fit
uv run soundbench fit-banjo --help
```

Supply recordings you have permission to analyze in a local directory.
A notes JSON is a list of objects with `path`, `pitch` (MIDI) and an assumed physical `string` (1–5).
The audio loader records file hashes and measured tuning; inspect its onset alignment and the string assignments before fitting.
Paths in the list must point to your local files.

```json
[
  {"path": "references/audio/note-D3.wav", "pitch": 50, "string": 4},
  {"path": "references/audio/note-G3.wav", "pitch": 55, "string": 3}
]
```

These names illustrate the schema, not bundled recordings.
Use several registers and reserve notes/seeds for validation rather than fitting every available observation.

## Fit and inspect

```sh
uv run soundbench fit-banjo --notes notes.json --out build/my-fit --steps 200
```

The fitter writes the aligned inputs, optimization metrics, source hashes and diagnostic report under the output directory, and the fitted preset as `preset.toml`.
Play it with `uv run soundbench play banjo --preset build/my-fit/preset.toml --out build/my-fit.wav`.
A fit starts from the banjo's hand-set `membrane` preset.
For a native-MP3 recording-chain fit, read `--chain`, `--native-mp3` and the stage-ablation options in the help.
For a runtime-aligned fit, inspect `--deployable` and the report's export limitations.
No external reference recording is fetched by the fit command.

Compare held-out notes through the same estimated recording chain.
Inspect the attack, isolated upper-partial decay, pitch, body ringing, and residual spectrum separately.
Then compare the dry instruments at matched integrated loudness and ask a human listener.
A fitted parameter pinned to a bound can be a model mismatch rather than a measured physical fact.

The lower-level `soundbench.scripts.models.banjo_fit` modules expose the differentiable twin, spectral/envelope losses, audio alignment and recording-chain model.
Piano calibration utilities are under `soundbench.scripts.models.piano_fit` and related modules; they expect an explicit local sample-package installation and don't bundle it.

## Rights

Record a source's creator, licence, URL, checksum, analysis windows and any assumed fingering.
Keep those with every exported parameter set.
Source audio, aligned crops and figures derived from it are local outputs, not automatically redistributable artifacts.
See `soundbench/presets/PROVENANCE.toml` before publishing a fit.
