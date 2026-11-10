# Soundbench

A studio of instruments that a human or a language model can play.
Each instrument is a physical model in code (strings, bores, membranes, vocal tracts), so it synthesizes its own sound without samples or generative audio models.
Play it from Python notes, a MIDI file or the command line.
The same library carries the ears an agent uses in place of listening: spectrograms, loudness, pitch and timbre measurements, and tools that fit an instrument's parameters to recordings.

These instruments play every modelled part on the album *Era*; [the instrument table](docs/instruments.md) says which settings it uses and what each instrument's known flaws are.

```text
notes + articulations / MIDI → physical instrument → audio
                                                      ↓
                          spectra · loudness · pitch · recording fits
```

## Make sound

Install [uv](https://docs.astral.sh/uv/) and [FFmpeg](https://ffmpeg.org/).
On macOS, `brew install uv ffmpeg` supplies both; on Linux, install FFmpeg with your package manager and uv with its documented installer.
No soundfonts, model weights, notation software or reference recordings are needed for the quickstart.

```sh
git clone https://github.com/cailmdaley/soundbench.git
cd soundbench
uv sync
uv run soundbench play banjo --pitch 60 --out build/banjo.wav
```

The command renders three plucks at different velocities, including their decay.
Open `build/banjo.wav` in an audio player.
To play a MIDI file instead, use `uv run soundbench play piano --midi notes.mid --out build/piano.wav`.
All note-bearing tracks play the selected instrument; MIDI controllers and pitch bends carry its articulations.

```python
from soundbench import Note, render
import soundfile as sf

notes = [Note(60, start=0, duration=0.5, velocity=72),
         Note(67, start=0.75, duration=0.5, velocity=80)]
audio, sample_rate = render("guitar", notes)
sf.write("guitar.wav", audio, sample_rate, subtype="FLOAT")
```

Audio is a floating-point `[frames, channels]` array at 48 kHz, with no automatic normalization.
A renderer subprocess releases its compilation and audio buffers when each call finishes.
The CLI writes a floating-point WAV so synthesis peaks aren't silently clipped to integers.

## Instruments and tools

The library includes banjo, guitar, piano, upright/electric bass, fiddle, mandolin, dobro, acoustic and electronic drums, keyboards, winds, friction/percussion models, invented instruments, several tract singers, a machine voice, and a room/microphone/tape stage.
Each row of [the instrument table](docs/instruments.md) gives a probe command; the playing sheets under `references/` document each instrument's range, controls and gestures.

```sh
uv run soundbench spectrum build/banjo.wav --out build/banjo.png
uv run soundbench listen --help
uv run soundbench compare --help
uv run soundbench tempo --help
```

An instrument is its physical model; a parameter file is a preset, one setting of that model, set by hand or fitted to some recordings.
The library ships a default for each instrument, and the fitting tools let you reach your own from recordings you choose: install the fitting extra with `uv sync --extra fit` and see [the fitting guide](docs/fitting.md).
Reference audio is an explicit local input, never bundled or downloaded by a render.

[AGENTS.md](AGENTS.md) covers playing, developing and measuring instruments.
The library contains no album, original song collection, private listening room or recordings.

## Tests

```sh
uv sync --all-extras
uv run python scripts/test.py
```

Tests render diagnostic notes and can take several minutes.
The runner isolates test files so JIT caches don't accumulate across instruments.
Private-studio delivery checks don't belong to this library; any retained check needing a private recording skips with its reason.

## Licence

MIT, copyright Cail Daley; see [LICENSE](LICENSE).
Soundbench was written by Claude with Cail Daley.
[RIGHTS.md](RIGHTS.md) credits the recordings and vendored code the library draws on.
