#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.2", "optax>=0.2,<0.3", "numpy>=2", "scipy>=1.14", "numba>=0.61", "soundfile>=0.13", "matplotlib>=3.9"]
# ///
"""Fit isolated notes jointly or independently and generate an evidence report."""
from __future__ import annotations
import argparse
import copy
import json
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from soundbench.scripts.models.banjo import load_config, render as engine_render
from soundbench.scripts.paths import preset_path

# A fit starts from the hand-set membrane design, not from an earlier fit.
CONFIG = preset_path('banjo', 'membrane')
from soundbench.scripts.models.banjo_fit import twin, loss, chain
from soundbench.scripts.models.banjo_fit.audio import restore_scales


def optimize(p, base, config, notes, target, steps, rate, flags, sparsity, metallic_weight=0., freeze_body=False, fit_body_gains=False, *, snapshot=None, snapshot_every=1, ridge_weight=0.):
    if snapshot_every < 1:
        raise ValueError('snapshot_every must be positive')
    def record(index, parameters, value):
        if snapshot is not None and (index % snapshot_every == 0 or index == steps):
            # Host copies keep even a mutating consumer outside optimizer state.
            snapshot(index, {key: np.array(v, copy=True) for key, v in parameters.items()}, value)
    if flags.get('deployable'):
        p = twin.canonical_excitation(p, base, config, notes)
    prepared = loss.prepare(jnp.array(target), notes)
    metallic = loss.metallic_prepare(jnp.array(target), notes) if metallic_weight else None
    ridge = loss.ridge_prepare(jnp.array(target), notes) if ridge_weight else None
    optimizer = optax.chain(optax.clip_by_global_norm(10.), optax.adam(rate))
    state = optimizer.init(p)
    frequencies = np.asarray(base[0])*np.exp(.25*np.tanh(np.asarray(p['mode_frequency'])))
    protected = jnp.array([int(np.argmin(abs(frequencies-f))) for f in (1200., 3600.)])
    def objective(parameters):
        predicted = twin.render(parameters, base, config, notes, target.shape[-1], **flags)
        value = loss.objective(predicted, prepared, parameters, base, sparsity)
        if ridge_weight:
            value += ridge_weight*loss.ridge_term(predicted, ridge)
        if 'chain_mic' in parameters:
            value += chain.penalty(parameters, **flags)
        return value + metallic_weight*loss.metallic_term(predicted, metallic) if metallic_weight else value
    @jax.jit
    def step(parameters, state):
        value, gradient = jax.value_and_grad(objective)(parameters)
        if freeze_body:
            for key in ('mode_frequency', 'mode_time'):
                gradient[key] = jnp.zeros_like(gradient[key])
            if fit_body_gains:
                gradient['mode_gain'] = gradient['mode_gain'].at[protected].set(0.)
            else:
                gradient['mode_gain'] = jnp.zeros_like(gradient['mode_gain'])
        updates, state = optimizer.update(gradient, state, parameters)
        return optax.apply_updates(parameters, updates), state, value
    history = [float(objective(p))]
    record(0, p, history[0])
    best, best_loss = p, history[0]
    # Eager diagnostics need not coexist with the compiled gradient executable.
    import gc
    jax.clear_caches()
    gc.collect()
    start = time.monotonic()
    for k in range(steps):
        updated, state, value = step(p, state)
        number = float(value)
        if not np.isfinite(number):
            raise FloatingPointError(f'nonfinite objective at step {k}')
        if k > 0:
            record(k, p, number)
        if number < best_loss:
            best, best_loss = p, number
        p = updated
        history.append(number)
        if k%20 == 0 or k == steps-1:
            print(f'{k+1}/{steps}: loss={number:.5f}; elapsed={time.monotonic()-start:.1f}s', flush=True)
    final_loss = float(objective(p))
    if steps > 0:
        record(steps, p, final_loss)
    if final_loss < best_loss:
        best = p
    history.append(final_loss)
    return best, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes', type=Path, required=True, help='JSON list: path, MIDI pitch, assumed string (1..5)')
    parser.add_argument('--snapshot-out', type=Path, help='Directory for side-channel NPZ optimizer states (not best-state selections)')
    parser.add_argument('--snapshot-every', type=int, default=1, help='Record every N states, including initial and final')
    parser.add_argument('--out', type=Path, default=Path('build/banjo-fit'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--modes', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=.015)
    parser.add_argument('--initial-b', type=float, default=5e-5, help='Additional stiffness seed, 0 < B < 0.02')
    parser.add_argument('--sparsity', type=float, default=.003)
    parser.add_argument('--no-glide', action='store_true')
    parser.add_argument('--no-noise', action='store_true')
    parser.add_argument('--two-stage', action='store_true')
    parser.add_argument('--deployable', action='store_true', help='Fit the exact FIR loop with shared per-string excitation and fixed RMS')
    parser.add_argument('--chain', action='store_true', help='Enable all four shared recording stages')
    for stage in ('room', 'mic', 'background', 'codec'):
        parser.add_argument('--no-'+stage, action='store_true', help='Disable this recording stage')
    parser.add_argument('--native-mp3', action='store_true', help='Float-decode source MP3 and measure pre-onset PSD')
    parser.add_argument('--polarization-start', action='store_true', help='Seed the exportable second polarization at 3 cents and 0.15 mix')
    parser.add_argument('--ridge-weight', type=float, default=0., help='Resolved nonharmonic mid-band excess penalty')
    parser.add_argument('--metallic-weight', type=float, default=0., help='Weight for attack/tail upper-partial relative envelopes')
    parser.add_argument('--freeze-body', action='store_true', help='Preserve every body mode, including controlled formant damping')
    parser.add_argument('--fit-body-gains', action='store_true', help='With freeze-body, fit gains except the 1.2/3.6 kHz formants')
    parser.add_argument('--independent', action='store_true', help='Fit one instrument per note instead of a shared instrument')
    parser.add_argument('--resume', type=Path, help='Start from parameters.json from a compatible fit')
    parser.add_argument('--allow-unverified-resume', action='store_true', help='Allow a seed without a saved configuration hash; not for report-only')
    parser.add_argument('--report-only', action='store_true', help='Regenerate diagnostics/export from --resume without optimization')
    parser.add_argument('--export', type=Path, help='Where to write the fitted preset; defaults to OUT/preset.toml')
    args = parser.parse_args()
    args.export = args.export or args.out/'preset.toml'
    if args.chain and (not args.native_mp3 or args.independent):
        parser.error('--chain requires --native-mp3 and a joint fit')
    if args.polarization_start and not args.two_stage:
        parser.error('--polarization-start requires --two-stage')
    if args.out.exists() and (args.out/'parameters.json').exists() and not args.report_only:
        parser.error('fit output must be fresh')
    if args.snapshot_every < 1 or (args.snapshot_out and (args.independent or args.report_only)):
        parser.error('snapshots require a joint optimization and a positive interval')
    if not 0 < args.seconds <= 2 or args.steps < 0 or args.modes < 2 or not 0 < args.initial_b < .02:
        parser.error('seconds must be 0..2, steps nonnegative, modes at least two')
    if args.ridge_weight < 0 or not np.isfinite(args.ridge_weight):
        parser.error('ridge-weight must be finite and nonnegative')
    if args.metallic_weight < 0 or not np.isfinite(args.metallic_weight):
        parser.error('metallic-weight must be finite and nonnegative')
    if args.fit_body_gains and not args.freeze_body:
        parser.error('fit-body-gains requires freeze-body')
    if args.independent and (args.metallic_weight or args.freeze_body):
        parser.error('metallic-weight and freeze-body require a joint fit')
    if args.freeze_body and not args.resume:
        parser.error('freeze-body requires a saved body via --resume')
    if args.report_only and (not args.resume or args.independent):
        parser.error('--report-only requires --resume and a joint fit')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'SOURCES.md').write_text(Path(__file__).with_name('SOURCES.md').read_text())
    # Audio parsing runs under -I; source files cannot shadow dependency imports.
    subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('audio.py')),
                    str(args.notes.resolve()), str((args.out/'aligned.npz').resolve()), str(args.seconds)]+(['--native'] if args.native_mp3 else []), check=True)
    data = np.load(args.out/'aligned.npz')
    target, notes = data['audio'], json.loads(str(data['notes']))
    config = load_config(args.config)
    p, base = twin.initial_parameters(config, notes, args.modes)
    p['b'] = jnp.full(5, np.log(args.initial_b/(.02-args.initial_b)))
    p.update(chain.initial_parameters())
    if args.polarization_start:
        p['second_mix'] = jnp.full(len(notes), np.log(.3/.7))
        p['second_detune'] = jnp.full(5, np.arctanh(3/8))
    flags = dict(glide=not (args.no_glide or args.deployable), noise=not (args.no_noise or args.deployable), two_stage=args.two_stage, deployable=args.deployable)
    flags.update({stage: args.chain and not getattr(args, 'no_'+stage) for stage in ('room', 'mic', 'background', 'codec')})
    resumed_notes = None
    if args.resume:
        resumed_notes = json.loads((args.resume.parent/'notes.json').read_text())
        previous = json.loads((args.resume.parent/'metrics.json').read_text())
        if previous['seconds'] != args.seconds:
            raise ValueError('resume duration differs from the saved fit')
        if args.report_only:
            flags = previous['flags']
            args.steps = previous['steps']
            args.initial_b = previous['initial_b']
            args.sparsity = previous['sparsity']
            args.learning_rate = previous['learning_rate']
            args.metallic_weight = previous.get('metallic_weight', 0.)
            args.ridge_weight = previous.get('ridge_weight', 0.)
            args.freeze_body = previous.get('freeze_body', False)
            args.fit_body_gains = previous.get('fit_body_gains', False)
        unverified = args.allow_unverified_resume and not args.report_only and previous.get('config_sha256') is None
        if not unverified and previous.get('config_sha256') != hashlib.sha256(args.config.read_bytes()).hexdigest():
            raise ValueError('resume configuration differs from the saved fit')
        if unverified:
            for note, old in zip(notes, resumed_notes):
                if any(note.get(k) != old.get(k) for k in ('path', 'pitch', 'string', 'sha256')):
                    raise ValueError('unverified seed recording manifest differs')
                old['decoded_sha256'] = note['decoded_sha256']
        restore_scales(notes, resumed_notes)
        saved = json.loads(args.resume.read_text())
        if np.shape(saved['loss'])[-1] != len(twin.KNOTS):
            old_knots = np.array(previous.get('decay_knots_hz', [80., 300., 1000., 3000., 8000., 24000.]))
            old_rates = np.array(previous['physical']['rates'])
            rates = np.array([np.interp(np.log(twin.KNOTS), np.log(old_knots), r) for r in old_rates])
            increments = np.diff(np.column_stack((np.full(5, twin.MIN_RATE), rates)), axis=1)
            saved['loss'] = np.log(np.maximum(increments, np.exp(-12))).tolist()
        p = {key: jnp.array(saved.get(key, value)) for key, value in p.items()}
        if any(p[key].shape != value.shape for key, value in twin.initial_parameters(config, notes, args.modes)[0].items()):
            raise ValueError('resume parameter shapes do not match this manifest/mode count')
    ancestry = []
    if args.resume:
        lineage_path = args.resume.parent/'lineage.json'
        if lineage_path.exists():
            parent = json.loads(lineage_path.read_text())
            ancestry = parent.get('resume_chain', [])+[dict(path=str(args.resume.resolve()), sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(), lineage=parent)]
        else:
            ancestry = [dict(path=str(args.resume.resolve()), sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(), ancestry='unrecorded')]
    lineage = dict(start='hand configuration' if not args.resume else 'saved seed', resume_chain=ancestry,
                   argv=sys.argv, flags=flags, config=str(args.config.resolve()),
                   config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   package_version=version('soundbench'),
                   versions=dict(jax=jax.__version__, numpy=np.__version__, optax=optax.__version__, python=sys.version),
                   recordings=[{k:n[k] for k in ('path','sha256','decoded_sha256','native_sha256') if k in n} for n in notes],
                   source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (args.out/'lineage.json').write_text(json.dumps(lineage, indent=2)+'\n')
    baseline = []
    for note in notes:
        matched_pitch = 69+12*np.log2(note['frequency']/440)
        row = [0, 0, note['string']-1, 0, matched_pitch, 90, 0, 0, 0, 0, 0]
        y = engine_render([row], args.seconds, config=config)[0]
        y *= .05/max(np.sqrt(np.mean(y**2)), 1e-10)
        baseline.append(y)
    baseline = np.array(baseline)
    # Match initial twin loudness once, leaving later level changes trainable.
    if resumed_notes is None:
        initial = np.array(twin.render(p, base, config, notes, target.shape[-1], **flags))
        for i, note in enumerate(notes):
            note['scale'] = float(.05/max(np.sqrt(np.mean(initial[i]**2)), 1e-10))
    if args.independent:
        histories, predictions, all_parameters, scores = [], [], [], []
        shared = {'loss', 'b', 'mode_frequency', 'mode_time', 'mode_gain', 'second_rate', 'second_detune', 'stages'}
        for i, note in enumerate(notes):
            local = {key: value if key in shared else value[i:i+1] for key, value in p.items()}
            fitted, history = optimize(local, base, config, [note], target[i:i+1], args.steps,
                                       args.learning_rate, flags, args.sparsity)
            predicted = np.array(twin.render(fitted, base, config, [note], target.shape[-1], **flags))
            predictions.append(predicted[0])
            spectral, envelope = loss.terms(predicted, loss.prepare(jnp.array(target[i:i+1]), [note]))
            scores.append(dict(pitch=note['pitch'], spectral=float(spectral), partial_envelope=float(envelope),
                               data_loss=float(spectral+.25*envelope), steps=args.steps))
            histories.append(history)
            all_parameters.append({key: np.asarray(v).tolist() for key, v in fitted.items()})
        (args.out/'independent-parameters.json').write_text(json.dumps(all_parameters, indent=2))
        np.savez(args.out/'independent-results.npz', target=target, fitted=np.array(predictions), baseline=baseline)
        (args.out/'loss.json').write_text(json.dumps(histories))
        (args.out/'independent-scores.json').write_text(json.dumps(scores, indent=2)+'\n')
        (args.out/'notes.json').write_text(json.dumps(notes, indent=2)+'\n')
        print('Independent fits saved; no shared-instrument export is defined.')
        return
    snapshot = None
    if args.snapshot_out:
        args.snapshot_out.mkdir(parents=True, exist_ok=True)
        def snapshot(index, parameters, value):
            np.savez_compressed(args.snapshot_out/f's{index:05d}.npz', step=index, objective=value, **parameters)
    if args.report_only:
        fitted = p
        history = json.loads((args.resume.parent/'loss.json').read_text())
    else:
        fitted, history = optimize(p, base, config, notes, target, args.steps, args.learning_rate, flags, args.sparsity,
                                   args.metallic_weight, args.freeze_body, args.fit_body_gains,
                                   snapshot=snapshot, snapshot_every=args.snapshot_every, ridge_weight=args.ridge_weight)
    if args.snapshot_out and not args.report_only:
        index = int(np.argmin(history))
        selected = max(0, index-1) if index < len(history)-1 else args.steps
        snapshot(selected, {k:np.asarray(v) for k,v in fitted.items()}, min(history))
    prediction = np.array(twin.render(fitted, base, config, notes, target.shape[-1], **flags))
    (args.out/'parameters.json').write_text(json.dumps({key: np.asarray(v).tolist() for key, v in fitted.items()}, indent=2))
    (args.out/'notes.json').write_text(json.dumps(notes, indent=2))
    (args.out/'loss.json').write_text(json.dumps(history))
    np.savez(args.out/'results.npz', target=target, baseline=baseline, fitted=prediction)
    from soundbench.scripts.models.banjo_fit.report import report
    report(args, config, notes, base, fitted, target, baseline, prediction, history, flags)


if __name__ == '__main__':
    main()
