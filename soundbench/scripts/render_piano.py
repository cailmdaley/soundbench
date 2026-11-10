#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy", "numba>=0.61", "scipy"]
# ///
"""Render plain MIDI piano notes/CC64/CC67 into source-attributed stereo stems."""
from collections import defaultdict, deque
from pathlib import Path
import argparse
import hashlib
import json
import time

import mido
import numpy as np

from soundbench.scripts.paths import parameters_label, parameters_path
from soundbench.scripts.models.piano import CONFIG, load_config, render
from soundbench.scripts.perform import TempoMap
from scipy.io.wavfile import write as write_pcm


def track_events(midi, index):
    tempo = TempoMap(midi)
    active = defaultdict(deque)
    notes, controls = [], []
    tick = 0
    for msg in midi.tracks[index]:
        tick += msg.time
        seconds = tempo.seconds_at(tick)
        if msg.type == 'note_on' and msg.velocity:
            row = {'time': seconds, 'off': None, 'pitch': msg.note, 'velocity': msg.velocity,
                   'channel': msg.channel}
            notes.append(row); active[msg.channel, msg.note].append(row)
        elif msg.type in ('note_off', 'note_on'):
            if active[msg.channel, msg.note]:
                active[msg.channel, msg.note].popleft()['off'] = seconds
        elif msg.type == 'control_change' and msg.control in (64, 67):
            controls.append((seconds, msg.control, msg.value, msg.channel))
    if any(active.values()):
        raise ValueError('piano MIDI has unterminated notes; every final note needs note-off')
    return notes, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    options = data.get('settings', {}).get('piano_model', {})
    if set(options)-{'parameters', 'output_gain'}:
        raise ValueError('unknown piano_model setting')
    root = CONFIG.parents[2]
    path = parameters_path(options.get('parameters', str(CONFIG.relative_to(root))))
    cfg = load_config(path)
    if 'output_gain' in options:
        gain = float(options['output_gain'])
        if not np.isfinite(gain) or not 0 < gain <= 4:
            raise ValueError('invalid piano output_gain')
        cfg['engine']['output_gain'] = gain
    midi = mido.MidiFile(data['performed_midi'])
    tempo = TempoMap(midi)
    end = max(sum(m.time for m in track) for track in midi.tracks)
    seconds = tempo.seconds_at(end)+cfg['engine']['tail_seconds']
    payload = path.read_bytes()+json.dumps(options, sort_keys=True).encode()
    for key in ('modal_data','release_data','coupled_data','body_data'):
        if key in cfg['engine']:
            payload+=(path.parent/cfg['engine'][key]).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    warm = time.perf_counter()
    render([{'time':0, 'off':.02, 'pitch':60, 'velocity':64}], .1, cfg)
    compile_seconds = time.perf_counter()-warm
    stems, elapsed, peak, count = {}, 0, 0, 0
    for job in data['jobs']:
        rows, controls = track_events(midi, job['index'])
        audio = np.zeros((round(seconds*48000), 2))
        start = time.perf_counter()
        # MIDI channels have independent pedals. Do not let channel 1's pedal
        # release channel 2's notes, even when they occupy the same MIDI track.
        for channel in sorted(set(r['channel'] for r in rows)):
            audio += render([r for r in rows if r['channel'] == channel], seconds, cfg,
                            [e[:3] for e in controls if e[3] == channel])
        elapsed += time.perf_counter()-start
        peak = max(peak, float(abs(audio).max(initial=0)))
        if not np.all(np.isfinite(audio)) or np.max(np.abs(audio), initial=0) >= 1:
            raise ValueError('piano stem is non-finite or clips; lower output_gain')
        write_pcm(Path(job['output']), 48000, np.rint(audio*2147483647).astype('<i4'))
        count += len(rows)
        stems[str(job['index'])] = {'engine':'model', 'model':'piano',
                                  'parameters':parameters_label(path), 'parameters_sha256':digest}
    metrics = {'stems':stems, 'notes':count, 'peak':peak, 'audio_seconds':seconds,
               'render_seconds':elapsed, 'compile_or_cache_seconds':compile_seconds,
               'realtime_factor':seconds*len(data['jobs'])/max(elapsed, 1e-9)}
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(f'Piano model: {count} notes, {seconds:.2f}s audio in {elapsed:.3f}s; peak {peak:.3f}')

if __name__ == '__main__':
    main()
