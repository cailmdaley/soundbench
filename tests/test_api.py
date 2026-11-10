"""Public notes/MIDI/CLI contract, including every registered instrument."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from soundbench import Note, notes_to_midi, render
from soundbench.api import RENDERERS

PITCHES = {name: 60 for name in RENDERERS}
PITCHES.update(bass=43, **{"electric-bass": 43, "fiddle": 69, "mandolin": 67, "dobro": 62,
                         "drums": 36, "tambori": 36, "ximbomba": 48, "flabiol": 84,
                         "tr808": 36, "tr808-bass": 36, "tr909": 36, "linn": 36})


@pytest.mark.parametrize("model", sorted(RENDERERS))
def test_each_instrument_makes_finite_sound(model):
    lyric = "ah" if model in {"voice", "voice2", "voice3", "chimera", "machine-voice"} else None
    audio, sr = render(model, [Note(PITCHES[model], 0, .35, 72, lyric)])
    assert sr == 48000
    assert audio.ndim == 2 and audio.shape[1] in (1, 2)
    assert len(audio) >= round(.35 * sr)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 1e-7


def test_midi_and_notes_agree():
    notes = [Note(60, 0, .3, 72), Note(64, .5, .3, 80)]
    direct, _ = render("banjo", notes)
    from_midi, _ = render("banjo", midi=notes_to_midi(notes))
    assert np.array_equal(direct, from_midi)


def test_controller_precedes_note_and_retrigger_follows_release():
    midi = notes_to_midi([Note(60, 0, .5), Note(60, .5, .5)], controls=[(0, 64, 127)])
    events = [m.type for m in midi.tracks[0] if not m.is_meta]
    assert events[:4] == ["control_change", "note_on", "note_off", "note_on"]


@pytest.mark.parametrize("note", [Note(60, -1, 1), Note(60, 0, 0), Note(60, float("nan"), 1), Note(128, 0, 1), Note(60, 0, 1, 0)])
def test_invalid_note_rejected(note):
    with pytest.raises(ValueError):
        notes_to_midi([note])


def test_cli_plays_notes_json(tmp_path):
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps([{"pitch": 60, "start": 0, "duration": .35}]))
    out = tmp_path / "note.wav"
    result = subprocess.run([sys.executable, "-c", "from soundbench.cli import main; main()", "play", "banjo", "--notes", str(notes), "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_every_parameter_has_provenance():
    import tomllib
    import soundbench
    bank = Path(soundbench.__file__).parent / "instruments/models"
    groups = tomllib.loads((bank / "PROVENANCE.toml").read_text())["sets"]
    names = [name for group in groups for name in group["files"]]
    assert len(names) == len(set(names))
    assert set(names) == {p.name for p in bank.iterdir() if p.suffix in {".toml", ".json"} and p.name != "PROVENANCE.toml"}
