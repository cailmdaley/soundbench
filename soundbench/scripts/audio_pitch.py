"""Audio-domain fundamental estimator for an isolated note of known score pitch."""
import numpy as np
from scipy import signal


def pitch_cents(audio, expected, sr=48000):
    """Band-limited autocorrelation with a sub-sample parabolic peak.

    The score supplies a ±4% search interval, not the answer: a maximum at the
    search boundary or silence is rejected. Use only isolated, settled notes.
    """
    f = 440 * 2 ** ((expected - 69) / 12)
    if len(audio) < sr/f*8 or np.max(np.abs(audio), initial=0) < 1e-12:
        raise ValueError("pitch measurement needs eight periods of audible audio")
    sos = signal.butter(4, (f*.7, f*1.3), fs=sr, btype="bandpass", output="sos")
    x = signal.sosfiltfilt(sos, audio)
    x -= x.mean()
    x *= signal.windows.hann(len(x))
    corr = signal.fftconvolve(x, x[::-1], mode="full")[len(x)-1:]
    left, right = max(1, int(np.floor(sr/f*.96))-1), int(np.ceil(sr/f*1.04))+2
    i = left + int(np.argmax(corr[left:right]))
    if i == left or i == right-1:
        raise ValueError("pitch lies outside the score-constrained search interval")
    d = .5 * (corr[i-1]-corr[i+1]) / (corr[i-1]-2*corr[i]+corr[i+1])
    if not sr/f*.96 <= i+d <= sr/f*1.04:
        raise ValueError("pitch lies outside the score-constrained search interval")
    return float(1200*np.log2((sr/(i+d))/f))


def isolated_pitch_report(audio, events, sr):
    rows = []
    for event in events:
        if len(event['pitches']) != 1 or event['end']-event['start'] < .65:
            continue
        start = event['start']+.10
        end = min(event['start']+.50,event['end'])
        try:
            cents = pitch_cents(audio[round(start*sr):round(end*sr)],event['pitches'][0],sr)
        except ValueError:
            continue
        rows.append({'start_s':event['start'],'midi':event['pitches'][0],'cents':cents})
    errors = [abs(r['cents']) for r in rows]
    return {'method':'Fundamental-band autocorrelation, parabolic lag interpolation; score-constrained ±4% search; isolated notes, 100–500 ms after onset',
            'count':len(rows),'max_abs_cents':float(max(errors)) if errors else None,
            'median_abs_cents':float(np.median(errors)) if errors else None,'notes':rows}
