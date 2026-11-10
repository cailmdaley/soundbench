#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "numba", "scipy", "cmudict", "mido"]
# ///
"""Intra-phrase re-articulation is continuous in controls and audio.

Run through the studio heavy runner; audio stays in memory.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from soundbench.scripts.models import phonetics, voice3
from soundbench.scripts.render_voice3 import apply_marks


KINDS = ('soft', 'breathy', 'hard', 'scooped')


def material(kind, attack_ms, marked):
    syl = phonetics.Syllable('a', [], ['AA1'], [])
    rows = [dict(time=.5 + i * 1.6, off=2.05 + i * 1.6, pitch=60, velocity=70, syllable=syl, tick=i*480)
            for i in range(2)]
    part = {'character': {'base': 'plain', 'onset': kind, 'attack_ms': attack_ms, 'vibrato': 'straight'}}
    marks = ([{'start_tick': 480, 'end_tick': 960, 'character': None,
               'patch': {'onset': kind, 'attack_ms': attack_ms}}] if marked else [])
    rows = apply_marks(rows, part, marks)
    c = voice3.load_config(character=part['character'], seed=7, jitter=0, shimmer=0,
                           flutter_cents=0, drift_cents=[0, 0], word_scoop_cents=[0, 0])
    return rows, c


def transient_ratio(y, reference, start=1.7, end=2.3):
    """Largest 1-ms high-pass residual RMS / steady vowel RMS.

    Subtract a phase/source-matched render whose amplitude lane is smoothed
    over 3 ms, isolating fast changes rather than intentional breath or colour.
    A 2-kHz high-pass retains broadband transients from abrupt modulation;
    the denominator comes from the preceding steady vowel, not a dip's floor.
    """
    residual = sosfilt(butter(4, 2000, 'highpass', fs=voice3.SR, output='sos'), y - reference)
    a, b = round(start * voice3.SR), round(end * voice3.SR)
    power = np.convolve(residual[a:b] ** 2, np.ones(48) / 48, mode='valid')
    steady = reference[round(1.2 * voice3.SR):round(1.6 * voice3.SR)]
    return float(np.sqrt(power.max() / np.mean(steady ** 2)))


class OnsetTests(unittest.TestCase):
    def test_every_intra_phrase_onset_has_bounded_per_ms_control_changes(self):
        for kind in KINDS:
            for marked in (False, True):
                for attack in (20, 70, 150):
                    with self.subTest(kind=kind, marked=marked, attack_ms=attack):
                        rows, c = material(kind, attack, marked)
                        k = voice3.controls(rows, c, 4)
                        window = (k['t'] >= 1.7) & (k['t'] <= 2.3)
                        env = k['envelope'][window]
                        amplitude = k['amplitude'][window]
                        self.assertLess(np.abs(np.diff(env)).max(), .08)
                        self.assertLess(np.abs(np.diff(amplitude)).max() / amplitude.max(), .08)
                        self.assertGreater(env.min(), 0)
                        self.assertAlmostEqual(env[0], 1)
                        self.assertAlmostEqual(env[-1], 1)

    def test_audio_high_pass_transients_are_bounded(self):
        for kind in KINDS:
            for marked in (False, True):
                with self.subTest(kind=kind, marked=marked):
                    rows, c = material(kind, 70, marked)
                    y = voice3.render(rows, 4, c)
                    controls = voice3.controls
                    def smoothed(*args, **kwargs):
                        k = controls(*args, **kwargs)
                        k['amplitude'] = gaussian_filter1d(k['amplitude'], 3)
                        return k
                    with patch.object(voice3, 'controls', side_effect=smoothed):
                        reference = voice3.render(rows, 4, c)
                    ratio = transient_ratio(y, reference)
                    print(f'onset {kind} marked={marked}: high-pass transient {ratio:.6f}', flush=True)
                    self.assertLess(ratio, .003)

    def test_high_pass_measure_rejects_a_source_amplitude_step(self):
        rows, c = material('soft', 70, False)
        controls = voice3.controls
        def stepped(*args, **kwargs):
            k = controls(*args, **kwargs)
            # Plan time includes the renderer's half-second pre-roll.
            age = k['t'] - (2.03 + c['v3']['engine']['lead_seconds'])
            active = (age >= 0) & (age < .07)
            k['amplitude'][active] *= voice3._ramp(age[active] / .07)
            return k
        def smoothed_step(*args, **kwargs):
            k = stepped(*args, **kwargs)
            k['amplitude'] = gaussian_filter1d(k['amplitude'], 3)
            return k
        with patch.object(voice3, 'controls', side_effect=stepped):
            bad = voice3.render(rows, 4, c)
        with patch.object(voice3, 'controls', side_effect=smoothed_step):
            reference = voice3.render(rows, 4, c)
        self.assertGreater(transient_ratio(bad, reference), .003)


if __name__ == '__main__':
    unittest.main()
