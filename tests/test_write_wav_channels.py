"""Model WAV output preserves duration and channel identity."""
import tempfile
import wave
from pathlib import Path
import numpy as np
import pytest
from soundbench.scripts.render_model import write_wav


def frames(audio):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "x.wav"
        write_wav(path, audio, 48000)
        with wave.open(str(path)) as stream:
            assert stream.getnchannels() == 2
            return stream.getnframes(), np.frombuffer(stream.readframes(stream.getnframes()), "<i4").reshape(-1, 2)


def test_mono_duration_and_channels():
    count, mono = frames(np.linspace(-.5, .5, 4800))
    assert count == 4800
    assert np.array_equal(mono[:, 0], mono[:, 1])


def test_stereo_duration_and_identity():
    stereo = np.stack([np.full(4800, .25), np.full(4800, -.25)], axis=1)
    count, output = frames(stereo)
    assert count == 4800
    assert (output[:, 0] > 0).all() and (output[:, 1] < 0).all()


def test_reject_three_channels():
    with pytest.raises(ValueError):
        frames(np.zeros((10, 3)))
