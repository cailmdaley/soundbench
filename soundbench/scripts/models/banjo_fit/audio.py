"""Isolated-process recording ingestion; run with python -I audio.py MANIFEST OUT SECONDS."""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, find_peaks, welch
import subprocess
from math import gcd


def load(manifest, seconds, native=False):
    path = Path(manifest).resolve()
    notes = json.loads(path.read_text())
    output = []
    for note in notes:
        source = (path.parent / note['path']).resolve()
        with source.open('rb') as stream:
            note['decoded_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if native:
            mp3 = source.with_suffix('.mp3')
            note['native_sha256'] = hashlib.sha256(mp3.read_bytes()).hexdigest()
            if note['native_sha256'] != note['sha256']:
                raise ValueError(f'Native source hash mismatch: {mp3}')
            y = np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(mp3),
                             '-f', 'f32le', '-ar', '48000', '-ac', '1', 'pipe:1']), dtype='<f4').astype(float)
            sr = 48000
        else:
            y, sr = sf.read(source, always_2d=True)
            y = y.mean(axis=1)
        if sr != 48000:
            divisor = gcd(sr, 48000)
            y = resample_poly(y, 48000//divisor, sr//divisor)
        y -= np.mean(y)
        # Retain 5 ms before the first strong attack; low recording noise isn't an onset.
        frames = np.pad(y, (0, (-len(y))%48)).reshape(-1, 48)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        onset = max(0, int(np.flatnonzero(rms > .15*rms.max())[0])*48-240)
        pre = y[960:max(960, onset-1920)]
        if native and len(pre) < 1024:
            raise ValueError(f'Insufficient pre-onset data: {source}')
        if native:
            freq, psd = welch(pre, 48000, nperseg=min(4096, len(pre)), scaling='density')
            anchors = np.geomspace(10., 24000., 24)
            note['pre_onset_psd'] = np.exp(np.interp(np.log(anchors), np.log(np.maximum(freq, 1)), np.log(psd+1e-30))).tolist()
        y = y[onset:]
        note['source_samples'] = len(y)
        note['onset_seconds'] = onset/48000
        nominal = 440*2**((note['pitch']-69)/12)
        segment = y[4800:min(len(y), 33600)]
        nfft = 2**18
        spectrum = abs(np.fft.rfft(segment*np.hanning(len(segment)), nfft))
        frequencies = np.fft.rfftfreq(nfft, 1/48000)
        estimates = []
        for partial in range(1, 5):
            candidates = np.flatnonzero((frequencies > nominal*partial*.95)&(frequencies < nominal*partial*1.05))
            peak = candidates[np.argmax(spectrum[candidates])]
            if spectrum[peak] < .02*spectrum.max():
                continue
            log = np.log(spectrum[peak-1:peak+2]+1e-20)
            shift = .5*(log[0]-log[2])/(log[0]-2*log[1]+log[2])
            estimates.append(((peak+shift)*48000/nfft/partial, float(spectrum[peak]), partial))
        if not estimates:
            raise ValueError(f'No reliable partials in {source}')
        # Prefer the fundamental when audible; otherwise the two lowest available partials.
        selected = estimates[:1] if estimates[0][2] == 1 else estimates[:2]
        note['frequency'] = float(np.average([x[0] for x in selected], weights=[x[1] for x in selected]))
        note['tuning_partials'] = [x[2] for x in selected]
        note['tuning_cents'] = float(1200*np.log2(note['frequency']/nominal))
        count = round(seconds*48000)
        y = np.pad(y[:count], (0, max(0, count-len(y))))
        note['rms_before_alignment'] = float(np.sqrt(np.mean(y**2)))
        y *= .05/max(note['rms_before_alignment'], 1e-10)
        output.append(y)
    if native:
        shared = np.median([n['pre_onset_psd'] for n in notes], axis=0).tolist()
        for note in notes:
            note['background_psd'] = shared
            note['background_scale'] = .05/max(note['rms_before_alignment'], 1e-10)
    return np.stack(output).astype(np.float32), notes


def restore_scales(notes, saved):
    """Keep the original loudness reference when resuming the same recordings."""
    if len(saved) != len(notes) or any(
            any(a.get(key) != b.get(key) for key in ('pitch', 'string', 'decoded_sha256'))
            for a, b in zip(saved, notes)):
        raise ValueError('resume notes differ from the saved recordings or assignments')
    for note, saved_note in zip(notes, saved):
        note['scale'] = saved_note['scale']


if __name__ == '__main__':
    y, notes = load(sys.argv[1], float(sys.argv[3]), '--native' in sys.argv[4:])
    np.savez(sys.argv[2], audio=y, notes=json.dumps(notes))
