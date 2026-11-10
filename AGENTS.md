# Play, build and measure instruments

Read `README.md` and `docs/instruments.md` first.
The instrument's `references/<name>/PLAYING.md` defines its sounding range, controllers and release behavior.
There are no bundled recordings or song scores.

## Play an instrument

Use `soundbench.render(model, notes)` or `render(model, midi=path)` from Python.
`preset="steel"` picks a named preset (`soundbench.presets(model)` lists them), and a path picks a preset file of your own.
A `Note` has MIDI pitch, start/duration in seconds, velocity and an optional lyric.
`controls=[(seconds, controller, value), ...]` adds CC events to a note list.
For pitch bends, multiple tracks or detailed controller curves, supply a MIDI file.
The output is floating-point audio at 48 kHz; it is not mastered or loudness-normalized.

`settings` accepts the instrument's model table and score-addressed articulation tables documented by the control sheet.
For note-list inputs, the notional meter is 4/4 at quarter = 120; MIDI inputs retain their own tempo/meter.
A bar/beat gesture is anchored to the matching performed note onset.
For custom timing, use MIDI rather than changing that note-list convention.

The command line accepts MIDI, a JSON array of note fields, and a TOML settings file:

```sh
uv run soundbench play guitar --midi notes.mid --settings guitar.toml --out build/guitar.wav
```

A singer needs lyrics attached to its notes; a note without a new lyric can extend the preceding vowel.
Keep phrases inside the singer's declared range.
Do not infer that high notes passing a metric sound pleasant.

## Develop an instrument

Synthesis lives under `soundbench/scripts/models/`; presets live under `soundbench/presets/<instrument>/`.
An instrument has one model and one folder: `default.toml`, named presets (a preset beginning `inherits = "default"` holds only what differs), and internal data in `data/`.
A different setting of the same physics is a preset, never a second model; recording chains (tape, microphones, rooms) live in `soundbench/rooms/`.
MIDI scheduling and articulation translation live in `soundbench/scripts/render_*.py`.
Register a new model in `soundbench/api.py` and the internal instrumentation/dispatch registries.
Keep one interface: note or MIDI events and settings in, finite floating-point audio out.

Add tests for tuning across the compass, velocity response, seed repeatability, release tails, controller changes, overlapping notes and replacement-note ownership.
A short diagnostic étude is welcome; a song or album doesn't belong here.
Document the compass, controls, approximations and actual listening status.
Add every preset and data file to `soundbench/presets/PROVENANCE.toml`, crediting any openly licensed recordings it was fitted to.
Never add a sample, reference excerpt or transcription merely because fitting used it.
A fitted preset is a setting of its model, not the model of the recorded instrument; name and describe it by how it plays.
Ship only presets someone would play: a fit's intermediate steps stay in the fit's output folder.

## Check by measurement

Render velocity ramps, repeated notes and isolated gestures before judging a full texture.
Use A-weighted spectrograms, pitch estimates, attack/decay measurements and section levels.
Read the actual plot and samples, including the final release.
A persistent horizontal spectral line may be a body resonance or a repeated note; inspect the events before deciding.

Compare alternatives within 0.5 LU of integrated loudness.
Keep one static gain per clip when testing dynamics or decay.
Separate instrument physics from microphone, room, background and codec effects when fitting a recording.
Use held-out notes/seeds, and distinguish source construction facts from assumptions.
A diagnostic is not a pass/fail gate until it separates known positive and negative controls.
Human listening decides timbre and musical feel.

Install all test extras with `uv sync --all-extras` and run `uv run python scripts/test.py`.
The runner isolates each file so JIT caches and score-helper imports don't accumulate across instruments.
Use focused `uv run pytest tests/test_<name>.py` checks during development.
Run one heavy process at a time; use the host studio's shared resource runner when it provides one.
Consequential audio changes need a fresh-context review from another model family.

## Boundaries

Keep personal feedback, original songs, delivery tooling, downloaded audio, model weights and generated files out of the repository.
Use explicit local input paths for recordings and write outputs under ignored `build/`.
The private composing studio consumes a pinned dependency; changes belong here, not in a second copy there.
