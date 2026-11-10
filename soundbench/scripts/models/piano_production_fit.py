#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "numba", "soundfile", "matplotlib"]
# ///
"""Fit a bounded physical piano and separable EQ/room/compressor to note clips.

Input JSON is a list of {audio, onset, offset, pitch, velocity} records.
Outputs containing audio observations live only in --out; --preset and --chain
contain parameters only. Short clips cannot identify late decay, pedal or room
independently: the report records these limitations and active bounds.
"""
import argparse
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from numba import njit
from scipy.optimize import least_squares
from scipy.signal import stft, find_peaks, resample_poly, sosfilt, butter, fftconvolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundbench.scripts.models.piano import CONFIG, coefficients, load_config, render, validate_adjustments

FS = 12000
BANDS = np.array([250., 1500., 5000.])


def mono(y):
    return y.mean(axis=1) if y.ndim == 2 else y


def spectrum(y, size=1024, hop=128):
    f, t, z = stft(y, FS, nperseg=size, noverlap=size-hop, boundary='zeros')
    return f, t, abs(z)


def estimate_partials(y, sr, pitch):
    x = y[round(.06*sr):round(min(.8, len(y)/sr)*sr)]
    if len(x) < 100:
        raise ValueError('target too short for frequency measurement')
    nfft = max(262144, 2**int(np.ceil(np.log2(len(x)))))
    mag = abs(np.fft.rfft(x*np.hanning(len(x)), nfft))
    freqs = np.fft.rfftfreq(nfft, 1/sr)
    peaks, _ = find_peaks(mag)
    nominal = 440*2**((pitch-69)/12)
    observations = []
    for k in range(1, 9):
        if nominal*k > sr*.44:
            break
        candidates = peaks[(freqs[peaks] > nominal*k*.975) & (freqs[peaks] < nominal*k*1.12)]
        if not len(candidates):
            continue
        p = candidates[np.argmax(mag[candidates])]
        if mag[p] < mag.max()*.008:
            continue
        # Quadratic log-peak interpolation reduces zero-padding bin error.
        v = np.log(mag[p-1:p+2]+1e-15)
        delta = .5*(v[0]-v[2])/(v[0]-2*v[1]+v[2])
        observations.append((k, (p+delta)*sr/nfft, mag[p]))
    if not observations:
        raise ValueError('no resolved partial')
    k, hz, amp = np.array(observations).T
    def residual(x):
        predicted = x[0]*k*np.sqrt((1+x[1]*k*k)/(1+x[1]))
        return (predicted-hz)*np.sqrt(amp/amp.max())
    fit = least_squares(residual, [hz[0]/k[0], .001], bounds=([nominal*.97, 0], [nominal*1.03, .02]), loss='soft_l1')
    # One carrier cannot identify B; retain a prior rather than claim a fit.
    if len(k) == 1:
        fit.x[:] = [hz[0], .001]
    unisons = peaks[(freqs[peaks] > fit.x[0]*.99) & (freqs[peaks] < fit.x[0]*1.01)]
    unisons = unisons[mag[unisons] > max(mag[unisons].max()*.04, 1e-12)] if len(unisons) else unisons
    unisons = unisons[np.argsort(mag[unisons])[-3:]]
    spacing = 1200*np.diff(np.log2(np.sort(freqs[unisons]))) if len(unisons)>1 else np.array([])
    spacing = spacing[(spacing >= .5) & (spacing <= 12)]
    return {'resolved_unison_hz': sorted(freqs[unisons].tolist()), 'unison_spacing_cents': float(np.median(spacing)) if len(spacing) else None, 'f0_hz': float(fit.x[0]), 'tuning_cents': float(1200*np.log2(fit.x[0]/nominal)), 'B': float(fit.x[1]), 'B_identified': len(k)>1, 'partial_hz': hz.tolist(), 'partial_numbers': k.astype(int).tolist()}


def features(y, frequencies):
    f, t, s = spectrum(y, 2048, 120)
    frequencies = [hz for hz in frequencies if 0 < hz < FS*.45]
    partial = np.array([np.sqrt(np.sum(s[np.abs(f-hz) <= max(12, hz*.003)]**2, axis=0)) for hz in frequencies])
    def slopes(lo, hi):
        mask = (t >= lo) & (t <= hi)
        if mask.sum() < 4:
            return [None]*len(partial)
        return [float(np.polyfit(t[mask], 20*np.log10(p[mask]+1e-9), 1)[0]) for p in partial]
    attack = s[:, t <= .05].sum(axis=1)
    # Frequency-domain IEC A-weighting magnitude, used only for diagnostics/loss.
    f2 = f*f
    ra = (12200**2*f2*f2)/((f2+20.6**2)*np.sqrt((f2+107.7**2)*(f2+737.9**2))*(f2+12200**2))
    envelope = 10*np.log10(np.sum((s*ra[:, None])**2, axis=0)+1e-14)
    between = (f>=1000) & (f<=5000)
    for hz in frequencies:
        between &= abs(f-hz)>max(30, hz*.007)
    interpartial = 10*np.log10(np.mean(s[between]**2, axis=0)+1e-14)
    band_energy = [10*np.log10(np.mean(s[(f>=lo)&(f<hi)]**2)+1e-14) for lo, hi in zip(np.geomspace(50, 5500, 17)[:-1], np.geomspace(50, 5500, 17)[1:])]
    floor = float(np.percentile(20*np.log10(s+1e-12), 20))
    return {'times': t, 'partial_db': 20*np.log10(partial+1e-9), 'early_db_s': slopes(.12, min(.7, len(y)/FS-.05)), 'late_window_db_s': slopes(.8, len(y)/FS-.05), 'attack_centroid_hz': float(np.sum(f*attack)/max(attack.sum(), 1e-15)), 'band_db': np.array(band_energy), 'a_envelope_db': envelope, 'interpartial_db': interpartial, 'spectral_floor_db': floor}


def eq_biquad(freq, gain_db, q, kind, sr=FS):
    """RBJ peaking and shelving EQ coefficients (one SOS row)."""
    a = 10**(gain_db/40)
    w = 2*np.pi*freq/sr
    c, s = np.cos(w), np.sin(w)
    alpha = s/(2*q)
    if kind == 'peak':
        b = np.array([1+alpha*a, -2*c, 1-alpha*a])
        d = np.array([1+alpha/a, -2*c, 1-alpha/a])
    else:
        z = 2*np.sqrt(a)*alpha
        if kind == 'low':
            b = a*np.array([(a+1)-(a-1)*c+z, 2*((a-1)-(a+1)*c), (a+1)-(a-1)*c-z])
            d = np.array([(a+1)+(a-1)*c+z, -2*((a-1)+(a+1)*c), (a+1)+(a-1)*c-z])
        elif kind == 'high':
            b = a*np.array([(a+1)+(a-1)*c+z, -2*((a-1)+(a+1)*c), (a+1)+(a-1)*c-z])
            d = np.array([(a+1)-(a-1)*c+z, 2*((a-1)-(a+1)*c), (a+1)-(a-1)*c-z])
        else:
            raise ValueError('invalid EQ type')
    return np.r_[b/d[0], d/d[0]]


@njit(cache=True)
def compress(y, threshold, ratio, attack, release, sr):
    result = np.empty_like(y)
    env, gain = 0., 0.
    aa, ar = np.exp(-1/(sr*attack)), np.exp(-1/(sr*release))
    for i in range(len(y)):
        v = abs(y[i])
        a = aa if v > env else ar
        env = a*env+(1-a)*v
        desired = min(0., (threshold-20*np.log10(env+1e-12))*(1-1/ratio))
        a = aa if desired < gain else ar
        gain = a*gain+(1-a)*desired
        result[i] = y[i]*10**(gain/20)
    return result


@lru_cache(maxsize=128)
def room_ir(sr, seed, rt60):
    """A deterministic three-band synthetic room, independent of input audio."""
    if sr != FS:
        # Sample one continuous band-limited room, rather than a different
        # random FIR at each rate. Scaling preserves its transfer gain in Hz.
        ir = resample_poly(room_ir(FS, seed, rt60), sr, FS)*(FS/sr)
        ir.flags.writeable = False
        return ir
    n = round(sr*2.)
    t = np.arange(n)/sr
    rng = np.random.default_rng(seed)
    ir = np.zeros(n)
    for low, high, rt in zip([60, 500, 2000], [500, 2000, min(5500, sr*.45)], rt60):
        noise = sosfilt(butter(2, [low, high], fs=sr, btype='bandpass', output='sos'), rng.standard_normal(n))
        ir += noise*np.exp(-6.9078*t/rt)
    ir[:round(sr*.008)] = 0
    ir /= max(np.linalg.norm(ir), 1e-12)
    ir.flags.writeable = False
    return ir


def production(y, params, sr=FS):
    """EQ → seeded synthetic room → static input gain → compressor."""
    out = y.copy()
    for eq in params['eq']:
        out = sosfilt(np.array([eq_biquad(eq['hz'], eq['gain_db'], eq['q'], eq['kind'], sr)]), out)
    room = params['room']
    if room['wet'] > 0:
        # Fixed support and seed make the basis continuous as RT60 changes.
        ir = room_ir(sr, room['seed'], tuple(room['rt60_seconds']))
        out = (1-room['wet'])*out+room['wet']*fftconvolve(out, ir)[:len(out)]
    c = params['compressor']
    out *= 10**(params['input_gain_db']/20)
    return compress(out, c['threshold_dbfs'], c['ratio'], c['attack_seconds'], c['release_seconds'], sr)


def production_audio(audio, params, sr=48000):
    """Identical per-channel chain for isolated targets and sequence rendering."""
    if audio.ndim == 1:
        return production(audio, params, sr)
    return np.column_stack([production(audio[:, c], params, sr) for c in range(audio.shape[1])])


EQ_SPEC = [(120, 'low'), (500, 'peak'), (1500, 'peak'), (3200, 'peak'), (5000, 'high')]


def chain_from_vector(x):
    return {'schema': 1, 'order': ['eq', 'room', 'input_gain', 'compressor'], 'eq': [{'hz': hz, 'kind': kind, 'q': .707 if kind != 'peak' else 1., 'gain_db': float(g)} for (hz, kind), g in zip(EQ_SPEC, x[:5])], 'room': {'type': 'banded_exponential_noise', 'seed': 1729, 'wet': float(x[5]), 'rt60_seconds': list(map(float, x[6:9]))}, 'compressor': {'threshold_dbfs': float(x[9]), 'ratio': float(x[10]), 'attack_seconds': float(x[11]), 'release_seconds': float(x[12])}, 'input_gain_db': float(x[13])}


def loss_terms(reference, estimate, frequencies):
    terms = []
    for n in (256, 1024, 2048):
        _, _, a = spectrum(reference, n, max(64, n//8))
        _, _, b = spectrum(estimate, n, max(64, n//8))
        # Reference-defined floor prevents the optimizer chasing inaudible bins.
        floor = max(a.max()*1e-3, 1e-8)
        difference = np.log10(a+floor)-np.log10(b+floor)
        terms.append(difference.ravel()/np.sqrt(difference.size))
    a, b = features(reference, frequencies), features(estimate, frequencies)
    for key, scale in [('partial_db', .03), ('a_envelope_db', .04), ('band_db', .03), ('interpartial_db', .08)]:
        delta = np.clip(a[key]-b[key], -30, 30)
        terms.append(scale*delta.ravel()/np.sqrt(max(delta.size, 1)))
    anchor = (a['times']>=.08) & (a['times']<=.15)
    if anchor.any():
        ac = a['partial_db']-a['partial_db'][:, anchor].mean(axis=1)[:, None]
        bc = b['partial_db']-b['partial_db'][:, anchor].mean(axis=1)[:, None]
        delta = np.clip(ac-bc, -40, 40)
        terms.append(.06*delta.ravel()/np.sqrt(max(delta.size, 1)))
    return np.concatenate(terms)


def joint_loss(residuals, target_count):
    """Sum already-normalized feature losses, then average over targets.

    Averaging over spectrogram pixels a second time would shrink data loss
    relative to physical priors when the FFT resolution or clip length grows.
    """
    return float(np.dot(residuals, residuals)/target_count)


def instrument_audio(target, cfg):
    # Keep keys down throughout the cropped observation; key-release time is
    # unidentifiable here and must not be inferred from a ringing tail.
    duration = len(target['y'])/FS
    audio = render([{'time': 0, 'off': duration, 'pitch': target['pitch'], 'velocity': target['velocity']}], duration+.1, cfg)
    return audio[:len(target['y'])*4]


def produced_audio(signal, chain):
    """Use the same 48-kHz nonlinear DSP as the continuous MIDI renderer."""
    return resample_poly(mono(production_audio(signal, chain, sr=48000)), 1, 4)


def serialize_preset(config, path):
    tables = ('engine', 'body_settings', 'hammer', 'strings', 'damper', 'resonance', 'soft_pedal', 'adjustments')
    lines = ['# Bounded physical model fitted to isolated recording targets.']
    for table in tables:
        if table not in config:
            continue
        lines += ['', '['+table+']']
        for key, value in config[table].items():
            if isinstance(value, dict):
                continue
            lines.append(f'{key} = {json.dumps(value)}')
        for key, values in config[table].items():
            if isinstance(values, dict):
                lines += ['', '['+table+'.'+key+']']
                lines.extend(f'{json.dumps(str(k))} = {json.dumps(v)}' for k, v in values.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n')


def plot_target(target, model, out, index):
    y = target['y']
    f, t, a = spectrum(y, 1024, 120)
    _, _, b = spectrum(model, 1024, 120)
    adb, bdb = 20*np.log10(a+1e-8), 20*np.log10(b+1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, z, title in zip(axes, [adb, bdb, bdb-adb], ['Reference', 'Model + production', 'Model − reference (dB)']):
        difference = title.startswith('Model −')
        im = ax.pcolormesh(t, f, z, shading='auto', cmap='RdBu_r' if difference else 'magma', vmin=-20 if difference else adb.max()-80, vmax=20 if difference else adb.max())
        ax.set_yscale('log'); ax.set_ylim(50, 5500); ax.set_xlabel('Seconds'); ax.set_title(title)
        fig.colorbar(im, ax=ax, label='dB')
    axes[0].set_ylabel('Hz')
    fig.savefig(out/f'note-{index}-{target["pitch"]}.png', dpi=150); plt.close(fig)
    af, bf = features(y, target['measurement']['partial_hz']), features(model, target['measurement']['partial_hz'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for i, hz in enumerate(target['measurement']['partial_hz'][:4]):
        axes[0].plot(af['times'], af['partial_db'][i], label=f'{hz:.1f} Hz reference')
        axes[0].plot(bf['times'], bf['partial_db'][i], '--', label=f'{hz:.1f} Hz model')
    axes[1].plot(af['times'], af['a_envelope_db'], label='Reference')
    axes[1].plot(bf['times'], bf['a_envelope_db'], label='Model')
    for ax in axes:
        ax.set_xlabel('Seconds'); ax.set_ylabel('dB'); ax.legend(fontsize=8)
    axes[1].set_title('A-weighted energy envelope')
    fig.savefig(out/f'decay-{index}-{target["pitch"]}.png', dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('targets', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--preset', type=Path, required=True)
    ap.add_argument('--chain', type=Path, required=True)
    ap.add_argument('--max-notes', type=int, default=6)
    ap.add_argument('--base-preset', type=Path, default=CONFIG)
    ap.add_argument('--initial-chain', type=Path)
    ap.add_argument('--frequency-anchor', choices=['slow', 'dominant'], default='dominant')
    ap.add_argument('--production-evals', type=int, default=20)
    ap.add_argument('--instrument-rounds', type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = json.loads(args.targets.read_text())[:args.max_notes]
    cfg = copy.deepcopy(load_config(args.base_preset))
    previous = copy.deepcopy(cfg.get('adjustments', {}))
    targets = []
    for row in raw:
        audio, sr = sf.read(row['audio'], start=round(row['onset']*sf.info(row['audio']).samplerate), stop=round(row['offset']*sf.info(row['audio']).samplerate))
        source = mono(audio)
        measure = estimate_partials(source, sr, row['pitch'])
        y = resample_poly(source, FS, sr)
        stereo = {}
        if audio.ndim == 2 and audio.shape[1] == 2:
            powers = np.mean(audio**2, axis=0)
            stereo = {'left_right_energy_db': float(10*np.log10((powers[0]+1e-15)/(powers[1]+1e-15))), 'left_right_correlation': float(np.corrcoef(audio.T)[0, 1])}
        targets.append({**row, 'y': y, 'measurement': measure, 'stereo_observation': stereo})
    if not targets:
        raise ValueError('no targets')
    pitch_cents = {}
    for pitch in sorted({r['pitch'] for r in targets}):
        pitch_cents[str(pitch)] = float(np.clip(np.median([r['measurement']['tuning_cents'] for r in targets if r['pitch']==pitch]), -50, 50))
    ratios = [r['measurement']['B']/coefficients(r['pitch'], r['velocity'], cfg)[-1] for r in targets if r['measurement']['B_identified']]
    detunes = {}
    profiles = cfg['coupled']['profiles']
    for pitch in sorted({r['pitch'] for r in targets}):
        spacings = [r['measurement']['unison_spacing_cents'] for r in targets if r['pitch']==pitch and r['measurement']['unison_spacing_cents'] is not None]
        prior = np.interp(pitch, [p['pitch'] for p in profiles], [p['mistuning_cents'] for p in profiles])
        detunes[str(pitch)] = float(np.clip(np.median(spacings) if spacings else prior, .05, 12))
    weights = {str(p): previous.get('pitch_string_weights', {}).get(str(p), [1.]*(1 if p<31 else 2 if p<43 else 3)) for p in sorted({r['pitch'] for r in targets})}
    cfg['adjustments'] = {'frequency_anchor': args.frequency_anchor, 'attack_gain': previous.get('attack_gain', .1), 'partial_gain_db': previous.get('partial_gain_db', [0., 0., 0., 0.]), 'pitch_string_weights': weights, 'pitch_detune_cents': detunes, 'pitch_cents': pitch_cents, 'inharmonicity_scale': float(np.clip(np.median(ratios) if ratios else 1, .25, 4)), 'detune_scale': 1., 'brightness_db_octave': 0., 'velocity_brightness': 0., 'prompt_rate_scale': [1., 1., 1.], 'late_rate_scale': [1., 1., 1.]}
    # Initialize frequency-dependent losses from measured carrier slopes,
    # rather than asking EQ to conceal persistent upper-partial aftersound.
    slope_ratios = {'prompt': [[], [], []], 'late': [[], [], []]}
    for target in targets:
        source_features = features(target['y'], target['measurement']['partial_hz'])
        baseline_features = features(resample_poly(mono(instrument_audio(target, cfg)), 1, 4), target['measurement']['partial_hz'])
        valid_hz = [hz for hz in target['measurement']['partial_hz'] if hz < FS*.45]
        for kind, feature_key in [('prompt', 'early_db_s'), ('late', 'late_window_db_s')]:
            for index, (hz, observed, baseline) in enumerate(zip(valid_hz, source_features[feature_key], baseline_features[feature_key])):
                if observed is None or baseline is None or observed >= -2 or baseline >= -2:
                    continue
                if baseline_features['partial_db'][index].max() < baseline_features['partial_db'].max()-50:
                    continue
                band = int(np.argmin(abs(np.log(hz/BANDS))))
                slope_ratios[kind][band].append(observed/baseline)
    for kind in ('prompt', 'late'):
        cfg['adjustments'][kind+'_rate_scale'] = [float(np.clip(np.median(v), .25, 6)) if v else 1. for v in slope_ratios[kind]]
    initial = copy.deepcopy(cfg)
    x = np.array([0, 0, 0, 0, 0, .03, .3, .3, .2, -18, 1.5, .01, .15, 0.], float)
    if args.initial_chain:
        chain = json.loads(args.initial_chain.read_text())
        c = chain['compressor']
        x = np.array([eq['gain_db'] for eq in chain['eq']] + [chain['room']['wet']] + chain['room']['rt60_seconds'] + [c['threshold_dbfs'], c['ratio'], c['attack_seconds'], c['release_seconds'], chain['input_gain_db']])
    lower = np.array([-12]*5+[0, .08, .08, .08, -40, 1, .00005, .003, -40.])
    upper = np.array([12]*5+[.3, 1.2, 1.2, 1.2, -5, 6, .08, .8, 40.])
    cache = {}
    def dry(config):
        signals = []
        for target in targets:
            adjustment = copy.deepcopy(config['adjustments'])
            for name in ('pitch_cents', 'pitch_detune_cents', 'pitch_string_weights'):
                adjustment[name] = {str(target['pitch']): adjustment[name][str(target['pitch'])]}
            key = json.dumps({'adjustments': adjustment, 'body': config['body_settings'], 'hammer': config['hammer'], 'pitch': target['pitch'], 'velocity': target['velocity'], 'frames': len(target['y'])}, sort_keys=True)
            if key not in cache:
                cache[key] = instrument_audio(target, config)
            signals.append(cache[key])
        return signals
    history = []
    def objective(params, signals, priors=True):
        chain = chain_from_vector(params)
        residuals = [loss_terms(r['y'], produced_audio(s, chain), r['measurement']['partial_hz']) for r, s in zip(targets, signals)]
        if priors:
            residuals.append(.04*(params-x)/(upper-lower))
        return np.concatenate(residuals)
    signals = dry(cfg)
    # Initial gain is one global recording-level alignment, never note-wise.
    if not args.initial_chain:
        x[-1] = float(np.clip(10*np.log10(sum(np.mean(r['y']**2) for r in targets)/sum(np.mean(s**2) for s in signals)), -40, 40))
    for round_index in range(args.instrument_rounds+1):
        fit = least_squares(lambda p: objective(p, signals), x, bounds=(lower, upper), max_nfev=args.production_evals, diff_step=.02)
        x = fit.x
        score = joint_loss(objective(x, signals, False), len(targets))
        history.append({'round': round_index, 'loss': score, 'production_nfev': fit.nfev, 'production_success': bool(fit.success)})
        print(history[-1], flush=True)
        if round_index == args.instrument_rounds:
            break
        # Alternating bounded coordinate search is joint fitting without letting
        # an opaque production filter impersonate an unbounded instrument.
        for attack_gain in (0., .03, .1, .3, 1.):
            trial = copy.deepcopy(cfg); trial['adjustments']['attack_gain'] = attack_gain
            test = dry(trial)
            trial_score = joint_loss(objective(x, test, False), len(targets))
            if trial_score < score:
                cfg, signals, score = trial, test, trial_score
        for name, values in [('noise_gain', [0., .05, .1, .25, .5, 1.]), ('decay_scale', [1., 2., 4., 6.])]:
            for value in values:
                trial = copy.deepcopy(cfg); trial['body_settings'][name] = value
                test = dry(trial)
                trial_score = joint_loss(objective(x, test, False), len(targets))
                if trial_score < score:
                    cfg, signals, score = trial, test, trial_score
        for scale in (1., 1.5, 1.8, 2., 2.3):
            trial = copy.deepcopy(cfg); trial['adjustments']['inharmonicity_scale'] = scale
            test = dry(trial)
            trial_score = joint_loss(objective(x, test, False), len(targets))
            if trial_score < score:
                cfg, signals, score = trial, test, trial_score
        candidates = [('brightness_db_octave', [-1., 1., 3., 5.])]
        for key, values in candidates:
            for value in values:
                trial = copy.deepcopy(cfg); trial['adjustments'][key] = value
                test = dry(trial)
                # Refit EQ for each excitation tilt: scoring with a fixed EQ
                # would double-count brightness and suppress the physical fit.
                trial_fit = least_squares(lambda p: objective(p, test), x, bounds=(lower, upper), max_nfev=max(4, args.production_evals//2), diff_step=.02)
                trial_score = joint_loss(objective(trial_fit.x, test, False), len(targets)) + .002*(value/3)**2
                if trial_score < score:
                    cfg, signals, score, x = trial, test, trial_score, trial_fit.x
        for partial in (1, 2):
            for db in (0., 6., 12.):
                trial = copy.deepcopy(cfg); trial['adjustments']['partial_gain_db'][partial] = db
                test = dry(trial)
                trial_score = joint_loss(objective(x, test, False), len(targets)) + .002*(db/6)**2
                if trial_score < score:
                    cfg, signals, score = trial, test, trial_score
        for pitch, original in detunes.items():
            for value in sorted({max(.05, original*.7), min(12., original*1.4), 4., 7., 9., 10.}):
                trial = copy.deepcopy(cfg)
                trial['adjustments']['pitch_detune_cents'][pitch] = value
                test = dry(trial)
                trial_score = joint_loss(objective(x, test, False), len(targets)) + .002*np.log(value/max(original,.05))**2
                if trial_score < score:
                    cfg, signals, score = trial, test, trial_score
        for pitch in weights:
            for excitation in ([1., .2, 1.], [1., 2., 1.], [.3, 1., 2.], [2., 1., .3], [0., 1., 1.]):
                trial = copy.deepcopy(cfg); trial['adjustments']['pitch_string_weights'][pitch] = excitation[:len(weights[pitch])]
                test = dry(trial)
                trial_score = joint_loss(objective(x, test, False), len(targets))
                if trial_score < score:
                    cfg, signals, score = trial, test, trial_score
        for kind in ('prompt', 'late'):
            for band in range(3):
                for scale in (.7, 1.4, 2.5, 4., 6.):
                    trial = copy.deepcopy(cfg)
                    trial['adjustments'][kind+'_rate_scale'][band] = scale
                    test = dry(trial)
                    trial_score = joint_loss(objective(x, test, False), len(targets)) + .005*np.log(scale)**2
                    if trial_score < score:
                        cfg, signals, score = trial, test, trial_score
        for gain in (.6, 1.4):
            trial = copy.deepcopy(cfg); trial['body_settings']['gain'] = gain
            test = dry(trial)
            trial_score = joint_loss(objective(x, test, False), len(targets)) + .002*np.log(gain)**2
            if trial_score < score:
                cfg, signals, score = trial, test, trial_score
    validate_adjustments(cfg['adjustments'])
    chain = chain_from_vector(x)
    serialize_preset(cfg, args.preset)
    args.chain.parent.mkdir(parents=True, exist_ok=True)
    args.chain.write_text(json.dumps(chain, indent=2)+'\n')
    report = {'production_sample_rate': 48000, 'analysis_sample_rate': FS, 'targets': [], 'instrument_initial': initial['adjustments'], 'instrument_fitted': cfg['adjustments'], 'body_gain': cfg['body_settings']['gain'], 'body_settings': cfg['body_settings'], 'frequency_anchor': cfg['adjustments']['frequency_anchor'], 'loss_slope_initialization_ratios': slope_ratios, 'optimizer': history, 'production_bound_hits': [int(i) for i in range(len(x)) if min(abs(x[i]-lower[i]), abs(x[i]-upper[i])) < .001], 'limitations': ['Alternating local fit, not a global optimum or unique source/production decomposition. Excitation tilt vs EQ is not uniquely identifiable from one velocity.', 'Instrument brightness is a modal excitation-envelope tilt, not an identified hammer material exponent.', 'Cropped repeated notes include preceding-note aftersound; attack velocity is a nuisance assumption.', 'Late-window slopes on short clips are not late T60 identification.', 'Velocity brightness, hammer exponent, key release and sustain pedal are retained at priors: these targets cannot identify them.', 'Source L/R energy and coherence are measured per target but the fit is mono. Pitch-dependent pan and treble decorrelation are not fitted.', 'Unison spacing and string-state weights are fitted per pitch; symmetric frequency offsets remain an approximation, but every observable eigenmode can radiate.', 'Synthetic room is mono; stereo phase, chorus/doubling, saturation and tape/flutter are not modelled.', 'Attack/key thumps, sustained residual modes and 3–4 Hz modulation may remain; no source residual waveform is replayed.']}
    baseline_cfg = load_config()
    for i, (target, signal) in enumerate(zip(targets, signals)):
        model = produced_audio(signal, chain)
        sf.write(args.out/f'model-{i}.wav', model, FS, subtype='FLOAT')
        sf.write(args.out/f'reference-{i}.wav', target['y'], FS, subtype='FLOAT')
        plot_target(target, model, args.out, i)
        rf, mf = features(target['y'], target['measurement']['partial_hz']), features(model, target['measurement']['partial_hz'])
        entry = {k: v for k, v in target.items() if k != 'y'}
        baseline = produced_audio(instrument_audio(target, baseline_cfg), chain)
        base_features = features(baseline, target['measurement']['partial_hz'])
        old_b = coefficients(target['pitch'], target['velocity'], baseline_cfg)[-1]
        entry['instrument_changes'] = {'tuning_cents': cfg['adjustments']['pitch_cents'][str(target['pitch'])], 'B_prior': float(old_b), 'B_fitted': float(old_b*cfg['adjustments']['inharmonicity_scale']), 'detune_prior_cents': float(np.interp(target['pitch'], [p['pitch'] for p in profiles], [p['mistuning_cents'] for p in profiles])), 'detune_fitted_cents': cfg['adjustments']['pitch_detune_cents'][str(target['pitch'])], 'prompt_rate_scale': cfg['adjustments']['prompt_rate_scale'], 'late_rate_scale': cfg['adjustments']['late_rate_scale']}
        entry['baseline_same_chain_a_envelope_median_absolute_db'] = float(np.median(abs(rf['a_envelope_db']-base_features['a_envelope_db'])))
        entry['baseline_same_chain_partial_median_absolute_db'] = float(np.median(abs(rf['partial_db']-base_features['partial_db'])))
        entry['interpartial_noise_median_signed_db'] = float(np.median(mf['interpartial_db']-rf['interpartial_db']))
        entry['attack_peak_reference_db'] = float(rf['a_envelope_db'][rf['times']<=.06].max())
        entry['attack_peak_model_db'] = float(mf['a_envelope_db'][mf['times']<=.06].max())
        entry['a_envelope_median_absolute_db'] = float(np.median(abs(rf['a_envelope_db']-mf['a_envelope_db'])))
        entry['band_median_absolute_db'] = float(np.median(abs(rf['band_db']-mf['band_db'])))
        entry['partial_median_absolute_db'] = float(np.median(abs(rf['partial_db']-mf['partial_db'])))
        entry['analysis_partial_hz'] = [hz for hz in target['measurement']['partial_hz'] if hz < FS*.45]
        entry['reference_early_db_s'] = rf['early_db_s']; entry['model_early_db_s'] = mf['early_db_s']
        entry['reference_late_window_db_s'] = rf['late_window_db_s']; entry['model_late_window_db_s'] = mf['late_window_db_s']
        entry['attack_centroid_reference_hz'] = rf['attack_centroid_hz']; entry['attack_centroid_model_hz'] = mf['attack_centroid_hz']
        entry['spectral_floor_db'] = rf['spectral_floor_db']
        report['targets'].append(entry)
    (args.out/'fit-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(args.out/'fit-report.json')


if __name__ == '__main__':
    main()
