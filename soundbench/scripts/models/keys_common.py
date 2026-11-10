"""Validation and sample-accurate controls shared by the two keyboard engines."""
import math
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
from soundbench.scripts.paths import preset_path, read_preset

ROOT = Path(__file__).resolve().parents[2]
SR = 48000


def load_config(name, path=None):
    cfg = read_preset(path or preset_path(name))
    if cfg['engine']['sample_rate'] != SR:
        raise ValueError('keyboards require 48 kHz')
    for table in cfg.values():
        if isinstance(table, dict):
            for value in table.values():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError('non-finite keyboard parameter')
    if type(cfg['engine']['seed']) is not int or cfg['engine']['seed'] < 0:
        raise ValueError('keyboard seed must be a nonnegative integer')
    if not 0 < cfg['engine']['output_gain'] <= 4:
        raise ValueError('output_gain must be in (0, 4]')
    return cfg


def validate(notes, seconds, controls, low, high):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('render duration must be positive')
    for n in notes:
        if not low <= n['pitch'] <= high or int(n['pitch']) != n['pitch']:
            raise ValueError(f'keyboard range is MIDI {low}–{high}')
        if not 1 <= n['velocity'] <= 127:
            raise ValueError('note velocity must be 1–127')
        if n.get('off') is None or not all(math.isfinite(n[k]) for k in ('time', 'off')) or not 0 <= n['time'] < n['off'] <= seconds:
            raise ValueError('every note needs a finite note-off inside the render')
    for t, cc, v in controls:
        if not math.isfinite(t) or not 0 <= t <= seconds or type(cc) is not int or not 0 <= cc <= 127 or not 0 <= v <= 127:
            raise ValueError('invalid keyboard controller')


def curve(controls, cc, length, default, sr=SR):
    y = np.full(length, float(default))
    for t, c, value in sorted(controls):
        if c == cc:
            y[min(round(t*sr), length):] = value
    return y


def smooth(y, seconds, sr=SR):
    from scipy.signal import lfilter
    a = np.exp(-1/(seconds*sr))
    return lfilter([1-a], [1, -a], y, zi=[a*y[0]])[0]


def dc_block(y, sr=SR):
    return sosfilt(butter(2, 12, 'highpass', fs=sr, output='sos'), y, axis=0)


def finish(y, gain):
    y = dc_block(y)*gain
    if not np.all(np.isfinite(y)):
        raise ValueError('non-finite keyboard output')
    return y
