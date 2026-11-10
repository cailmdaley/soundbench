#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render shared fiddle/violí strings from a studio JSON job manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import mido
from soundbench.scripts.paths import preset_bytes, preset_path
from soundbench.scripts.instrumentation import matching_tracks
from soundbench.scripts.models.fiddle import load_config, render
from soundbench.scripts.models.fiddle_player import CALIBRATION
from soundbench.scripts.perform import TempoMap
from soundbench.scripts.render_model import write_wav

CC = {2: ('force', lambda x: .002+.998*x/127),
      4: ('speed', lambda x: .01+.49*x/127),
      8: ('position', lambda x: .04+.26*x/127),
      1: ('vibrato', lambda x: 35*x/127),
      5: ('slide_seconds', lambda x: .01+.99*x/127),
      25: ('vibrato_delay', lambda x: 2*x/127),
      26: ('vibrato_rate', lambda x: 4+4*x/127),
      27: ('grit', lambda x: x/127),
      16: ('slide', lambda x: -3*x/127)}


def schedule(midi, settings, indices, config):
    """CCs are channel-scoped; controls update held notes as well as new notes."""
    tempo = TempoMap(midi)
    names = {i: midi.tracks[i].name for i in indices}
    pins = {}
    for selector, string in settings.get('fiddle_strings', {}).items():
        matches = matching_tracks(selector, indices, names)
        if not matches or type(string) is not int or string not in range(1, 5):
            raise ValueError('fiddle_strings selectors must match; strings are 1..4')
        for index in matches:
            if index in pins:
                raise ValueError('duplicate fiddle string binding')
            pins[index] = string
    rows, channels, last = [], [], 0.
    for owner, index in enumerate(indices):
        states, active = {}, {}
        tick = 0
        for msg in midi.tracks[index]:
            tick += msg.time
            t = tempo.seconds_at(tick)
            last = max(last, t)
            if not hasattr(msg, 'channel'):
                continue
            state = states.setdefault(msg.channel, {})
            if msg.type == 'control_change':
                if msg.control in CC:
                    key, mapping = CC[msg.control]
                    value = mapping(msg.value)
                    state[key] = value
                    if key in ('force', 'speed', 'position'):
                        for (channel, pitch), note in active.items():
                            if channel == msg.channel:
                                note['controls'].append((t-note['time'], key, value))
                elif msg.control == 22:
                    if msg.value not in range(4):
                        raise ValueError('CC22 is 0 normal, 1 surface, 2 crunch, 3 raw')
                    state['bow_style'] = ('normal', 'surface', 'crunch', 'raw')[msg.value]
                elif msg.control == 23:
                    if msg.value not in range(3):
                        raise ValueError('CC23 is 0 auto, 1 phrase end, 2 continuing')
                    if msg.value == 0:
                        state.pop('phrase_end', None)
                    else:
                        state['phrase_end'] = msg.value == 1
                elif msg.control in (68, 20, 21):
                    key = {68: 'slur', 20: 'string', 21: 'chop'}[msg.control]
                    if key == 'string':
                        if msg.value not in range(5):
                            raise ValueError('CC20 is 0 auto or 1..4 string')
                        if msg.value:
                            state[key] = msg.value
                        else:
                            state.pop(key, None)
                    else:
                        state[key] = msg.value >= 64
            elif msg.type == 'note_on' and msg.velocity:
                key = (msg.channel, msg.note)
                if key in active:
                    active.pop(key)['off'] = t
                note = dict(time=t, off=t, pitch=msg.note, velocity=msg.velocity,
                            owner=owner, controls=[], **state)
                if index in pins:
                    note['string'] = pins[index]
                rows.append(note)
                channels.append((owner, msg.channel))
                active[key] = note
            elif msg.type in ('note_off', 'note_on'):
                note = active.pop((msg.channel, msg.note), None)
                if note is not None:
                    note['off'] = t
        if active:
            raise ValueError('fiddle MIDI has unreleased notes')
    # Performed timing leaves legato overlaps inside one voice: the next finger
    # lands a few ms before the last lifts. That's a hand-on, not a double stop,
    # so the earlier note ends where the next begins (which also keeps slurs).
    # The same holds for any notes pinned to one string, across tracks: a
    # string sounds one pitch, so a drone and the next melody note hand over.
    voices = {}
    for row, voice in zip(rows, channels):
        voices.setdefault(voice, []).append(row)
        if 'string' in row:
            voices.setdefault(('string', row['string']), []).append(row)
    for notes in voices.values():
        notes.sort(key=lambda r: r['time'])
        for a, b in zip(notes, notes[1:]):
            if b['time']-a['time'] > .015 and a['off'] > b['time']:
                a['off'] = b['time']
                a['controls'] = [c for c in a['controls'] if c[0] <= a['off']-a['time']]
    return rows, max([r['off'] for r in rows]+[last])+config['engine']['tail_seconds']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    args = ap.parse_args()
    data = json.loads(args.manifest.read_text())
    options = data['settings'].get('fiddle_model', {})
    if set(options)-{'preset', 'tuning', 'output_gain'}:
        raise ValueError('fiddle_model accepts preset, tuning, output_gain')
    path = preset_path('fiddle', options.get('preset'))
    config = load_config(path, options.get('tuning', 'standard'))
    if 'output_gain' in options:
        gain = float(options['output_gain'])
        if not 0 < gain < 100:
            raise ValueError('output_gain must be finite, positive and below 100')
        config['engine']['output_gain'] = gain
    midi = mido.MidiFile(data['performed_midi'])
    rows, seconds = schedule(midi, data['settings'], [j['index'] for j in data['jobs']], config)
    start = time.perf_counter()
    audio, diagnostic = render(rows, seconds, len(data['jobs']), config, diagnostics=True, trace=False, channels=1)
    elapsed = time.perf_counter()-start
    stems = {}
    for owner, job in enumerate(data['jobs']):
        write_wav(Path(job['output']), audio[owner])
        stems[str(job['index'])] = dict(engine='model', model='fiddle',
            parameters=str(path), parameters_sha256=hashlib.sha256(preset_bytes(path)+CALIBRATION.read_bytes()+json.dumps(options, sort_keys=True).encode()).hexdigest(), overrides=options)
    import numpy as np
    metrics = dict(stems=stems, audio_seconds=seconds, render_seconds=elapsed,
                   peak=float(np.max(np.abs(audio))), assignment=diagnostic['assignment'],
                   player_calibration_sha256=hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
                   player_decisions=diagnostic['player_decisions'])
    # Player decisions carry numpy scalars.
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2, default=lambda o: o.item())+'\n')
    print(f'Fiddle: {seconds:.2f}s in {elapsed:.2f}s')


if __name__ == '__main__':
    main()
