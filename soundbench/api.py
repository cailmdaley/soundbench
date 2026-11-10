"""One sample-free rendering interface for notes, MIDI and instrument settings."""
from dataclasses import dataclass
from pathlib import Path
import json
import copy
import math
import subprocess
import sys
import tempfile

import mido
import numpy as np
import soundfile as sf

from .scripts.model_dispatch import RENDERERS
from .scripts.paths import preset_names

SAMPLE_RATE = 48000


@dataclass(frozen=True)
class Note:
    """A sounding MIDI pitch with onset/duration in seconds and optional lyric."""
    pitch: int
    start: float
    duration: float
    velocity: int = 80
    lyric: str | None = None


def notes_to_midi(notes, controls=()):
    """Create a one-part MIDI; controls are (seconds, CC number, value) triples."""
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="instrument"))
    track.append(mido.MetaMessage("set_tempo", tempo=500000))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    events, lyrics = [], []
    for note in notes:
        if not (math.isfinite(note.start) and math.isfinite(note.duration) and note.start >= 0 and note.duration > 0):
            raise ValueError("note times must be finite, with start >= 0 and duration > 0")
        if not (0 <= note.pitch <= 127 and 1 <= note.velocity <= 127):
            raise ValueError("MIDI pitch must be 0..127 and velocity 1..127")
        start, end = round(note.start * 960), round((note.start + note.duration) * 960)
        if end <= start:
            raise ValueError("note duration is shorter than one MIDI tick")
        events.extend([(start, 2, mido.Message("note_on", note=note.pitch, velocity=note.velocity)),
                       (end, 0, mido.Message("note_off", note=note.pitch))])
        if note.lyric is not None:
            lyrics.append((start, mido.MetaMessage("lyrics", text=note.lyric)))
    for seconds, control, value in controls:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("controller time must be finite and nonnegative")
        events.append((round(seconds * 960), 1, mido.Message("control_change", control=control, value=value)))
    last = 0
    for tick, _, event in sorted(events, key=lambda row: (row[0], row[1])):
        track.append(event.copy(time=tick-last)); last = tick
    if lyrics:
        lyric_track = mido.MidiTrack(); midi.tracks.append(lyric_track)
        last = 0
        for tick, event in sorted(lyrics, key=lambda row: row[0]):
            lyric_track.append(event.copy(time=tick-last)); last = tick
    return midi


def presets(model):
    """The named presets bundled for an instrument, default first."""
    return preset_names("tr808" if model == "tr808-bass" else model)


def render(model, notes=None, *, preset=None, midi=None, settings=None, controls=()):
    """Return (float audio [frames, channels], 48000) without normalization.

    Supply notes OR a MIDI path/MidiFile. `preset` is a bundled preset's name
    (see `presets(model)`) or a path to a preset file of one's own. MIDI CC/pitch
    bends and the settings sidecar carry articulations. All note-bearing tracks
    play the selected model. A renderer subprocess isolates JIT compilation and
    releases its memory.
    """
    if model not in RENDERERS:
        raise ValueError(f"unknown model {model!r}; choose from {', '.join(RENDERERS)}")
    if (notes is None) == (midi is None):
        raise ValueError("supply notes or midi, not both")
    if midi is not None and controls:
        raise ValueError("put controllers in the supplied MIDI")
    performance = notes_to_midi(notes, controls) if midi is None else (midi if isinstance(midi, mido.MidiFile) else mido.MidiFile(midi))
    indices = [i for i, track in enumerate(performance.tracks) if any(m.type == "note_on" and m.velocity for m in track)]
    if not indices:
        raise ValueError("the input contains no sounding notes")
    settings = copy.deepcopy(settings or {})
    settings["instruments"] = {str(n + 1): {"model": model} for n in range(len(indices))}
    if preset is not None:
        if not presets(model):
            raise ValueError(f"{model} has no presets")
        key = ("tr808" if model == "tr808-bass" else model).replace("-", "_") + "_model"
        settings.setdefault(key, {})["preset"] = str(preset)
    with tempfile.TemporaryDirectory(prefix="soundbench-") as directory:
        root = Path(directory)
        midi_path = root / "notes.mid"; performance.save(midi_path)
        jobs = [{"index": i, "output": str(root / f"part-{i}.wav")} for i in indices]
        manifest = root / f"render.{model}.json"
        manifest.write_text(json.dumps({"model": model, "instrument": model, "performed_midi": str(midi_path), "score_midi": str(midi_path),
                                        "settings": settings, "jobs": jobs}))
        process = subprocess.run([sys.executable, "-m", "soundbench.scripts." + RENDERERS[model][0].stem, str(manifest)],
                                 text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip())
        stems = []
        for job in jobs:
            audio, sr = sf.read(job["output"], dtype="float64", always_2d=True)
            if sr != SAMPLE_RATE:
                raise ValueError(f"renderer returned {sr} Hz, expected {SAMPLE_RATE}")
            stems.append(audio)
        frames = max(len(a) for a in stems); channels = max(a.shape[1] for a in stems)
        mixed = np.zeros((frames, channels))
        for audio in stems:
            mixed[:len(audio)] += audio
        return mixed, SAMPLE_RATE
