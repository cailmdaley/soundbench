"""Diagnostic figures, auditable metrics and a contractive waveguide export."""
import json
import hashlib
from pathlib import Path
import copy
import math
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.optimize import least_squares
import soundfile as sf
from soundbench.scripts.models.banjo import _coefficients, render as engine_render
from soundbench.scripts.models.banjo_fit import twin, loss, chain


def export_config(path, config, q, notes, two_stage=False, deployable=False):
    """Carry decay and polarization; share excitation at the string level."""
    c = copy.deepcopy(config)
    errors = []
    c['strings']['decay_hz'] = twin.KNOTS.tolist()
    c['strings']['decay_t60'] = (np.log(1000)/q['rates']).tolist()
    if two_stage:
        c['strings']['polarization_mix'] = [float(np.median([q['second_mix'][i] for i, n in enumerate(notes) if n['string'] == s+1])) for s in range(5)]
        c['strings']['polarization_rate'] = q['second_rate'].tolist()
        c['strings']['polarization_cents'] = q['second_detune'].tolist()
    from soundbench.scripts.models.string_loss import loss_bank
    from soundbench.scripts.models.banjo import _rich_coefficients
    bank = loss_bank(c['strings'])
    all_stages = []
    for s in range(5):
        # Measure actual FIR log-decay error at partials of the assigned notes.
        squared_errors = []
        for note in notes:
            if note['string'] != s+1:
                continue
            pitch = 69+12*np.log2(note['frequency']/440)
            _, _, h = _rich_coefficients(float(pitch), 48000, c['strings']['dispersion'][s], bank, s)
            frequencies = note['frequency']*np.arange(1, min(41, int(10000/note['frequency'])+1))
            z = np.exp(-2j*np.pi*frequencies[:,None]*np.arange(len(h))/48000)
            actual = -note['frequency']*np.log(abs(z@h))
            expected = np.interp(np.log(frequencies), np.log(twin.KNOTS), q['rates'][s])
            squared_errors.extend((np.log(actual)-np.log(expected))**2)
        errors.append(float(np.sqrt(np.mean(squared_errors))))
        if deployable:
            all_stages.append(q['stages'][s].tolist())
            continue
        pitch = c['strings']['open_midi'][s]
        f0 = 440*2**((pitch-69)/12)
        frequencies = f0*np.arange(1, min(40, int(10000/f0))+1)
        w = 2*np.pi*frequencies/48000
        ratio = frequencies/f0
        integer, a, _ = _coefficients(float(pitch), 48000, config['strings']['loss_mix'][s], config['strings']['dispersion'][s], config['strings']['t60'][s])
        def ap(coefficient):
            return -2*np.arctan((1-coefficient)/(1+coefficient)*np.tan(w/2))
        mix = config['strings']['loss_mix'][s]
        baseline = -integer*w + ap(a) + ap(config['strings']['dispersion'][s]) + np.angle(1-mix+mix*np.exp(-1j*w))
        B = q['B'][s]
        stiff = np.sqrt(2*ratio**2*(1+B)/(1+np.sqrt(1+4*B*ratio**2*(1+B))))
        target = baseline - 2*np.pi*(stiff-ratio)
        # Compensate the FIR's phase, then fit allpass curvature to stiff partials.
        c['strings']['dispersion'][s] = config['strings']['dispersion'][s]
        _, _, h = _rich_coefficients(float(pitch), 48000, c['strings']['dispersion'][s], bank, s)
        grid = np.linspace(0, np.pi, 4097)
        fir_phase = np.unwrap(np.angle(np.exp(-1j*grid[:, None]*np.arange(len(h)))@h))
        def residual(stages):
            integer, a, _ = _rich_coefficients(float(pitch), 48000, c['strings']['dispersion'][s], bank, s, stages)
            phase = -integer*w + ap(a) + ap(c['strings']['dispersion'][s]) + np.interp(w, grid, fir_phase)
            for d in stages:
                phase += ap(d)
            return (phase-target)/ratio
        result = least_squares(residual, np.array([-.6, -.5, -.3]), bounds=(-.98, 0), max_nfev=80)
        all_stages.append(result.x.tolist())
    c['strings']['dispersion_stages'] = q['stages'].tolist() if deployable else all_stages
    if deployable:
        if two_stage:
            c['strings']['polarization_mix'] = [float(np.mean([q['second_mix'][i] for i, n in enumerate(notes) if n['string'] == s+1])) for s in range(5)]
        for name, values in [('pick_ms', q['width']*1000), ('lowpass_hz', q['corner']), ('pluck_position', q['position'])]:
            c['excitation'][name+'_strings'] = [float(np.mean([values[i] for i, n in enumerate(notes) if n['string'] == s+1])) for s in range(5)]
    c['excitation']['exponential_pick'] = 1
    c['excitation']['velocity_corner_exponent'] = 0.
    fs, ts, gs = q['frequency'], q['t60'], q['gain']
    fundamental = float(gs[np.argmin(fs[:-1])])
    c['head'] = dict(fundamental_hz=float(fs[0]), ratios=(fs[:-1]/fs[0]).tolist(),
                     weights=gs[:-1].tolist(), t60_seconds=ts[:-1].tolist())
    c['pot'] = dict(frequency_hz=float(fs[-1]), t60_seconds=float(ts[-1]), weight=float(gs[-1]/fundamental))
    c['pickups']['head'] *= fundamental
    c['excitation']['pick_ms'] = float(np.median(q['width'])*1000)
    c['excitation']['thumb_pick_ms'] = c['excitation']['pick_ms']
    c['excitation']['lowpass_hz'] = float(min(np.median(q['corner']), 23999))
    c['excitation']['pluck_position'] = float(np.median(q['position']))
    c['excitation']['position_jitter'] = 0.
    c['bridge']['coupling'] = c['bridge']['head_feedback'] = 0.
    # Peak protection leaves passage loudness calibration independent of stress chords.
    if deployable:
        c['engine'].update(peak_knee=.90, peak_ceiling=.98)
    opens = c['strings']['open_midi']
    chords = [opens] + [[pitch]*5 for pitch in range(max(opens), min(opens)+c['strings']['max_fret']+1)]
    peak = 0.
    for chord in chords:
        rows = [[0, 0, s, 0, pitch, 127, 0, 0, 0, 0, s] for s, pitch in enumerate(chord)]
        peak = max(peak, float(np.max(np.abs(engine_render(rows, 1.5, config=c)))))
    gain_before = c['engine']['output_gain']
    # Delivered output gain is calibrated by engine_evidence.py on the étude.
    calibration = dict(probe_peak_before=peak, output_gain_before=gain_before,
                       output_gain_after=c['engine']['output_gain'], probe_peak_after=peak,
                       policy='Diagnostic stress peak; use EBU R128 passage calibration for delivered gain')
    text = '# Fitted series-body projection; see docs/fitting.md.\n'
    for table, values in c.items():
        text += f'\n[{table}]\n'
        for key, value in values.items():
            text += f'{key} = {json.dumps(value)}\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return c, errors, calibration


def attack_centroid(y):
    segment = y[:2400]*np.hanning(min(2400, len(y)))
    spectrum = abs(np.fft.rfft(segment))
    frequency = np.fft.rfftfreq(len(segment), 1/48000)
    mask = (frequency>=80)&(frequency<=10000)
    return float(np.sum(frequency[mask]*spectrum[mask])/(np.sum(spectrum[mask])+1e-20))


def score(y, prepared):
    spectral, envelope = loss.terms(jnp.array(y), prepared)
    return dict(spectral=float(spectral), partial_envelope=float(envelope),
                data_loss=float(spectral+.25*envelope))


def report(args, config, notes, base, p, target, baseline, fitted, history, flags):
    out = args.out
    q = {key: np.asarray(value) for key, value in twin.physical(p, base, config).items()}
    prepared = loss.prepare(jnp.array(target), notes)
    metrics = dict(current=score(baseline, prepared), fitted=score(fitted, prepared), flags=flags,
                   config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   steps=args.steps, seconds=args.seconds, initial_b=args.initial_b, sparsity=args.sparsity,
                   learning_rate=args.learning_rate, source_manifest=str(args.notes),
                   ridge_weight=getattr(args, 'ridge_weight', 0.), metallic_weight=args.metallic_weight, freeze_body=args.freeze_body, fit_body_gains=args.fit_body_gains,
                   decay_knots_hz=twin.KNOTS.tolist(), fft_samples=twin.fft_size(args.seconds), physical={key: value.tolist() for key, value in q.items()})
    ablations = {}
    for name, flag in [('without_glide', 'glide'), ('without_noise', 'noise'), ('without_second_stage', 'two_stage')]:
        if flags[flag]:
            changed = dict(flags, **{flag: False})
            predicted = np.array(twin.render(p, base, config, notes, target.shape[-1], **changed))
            ablations[name] = score(predicted, prepared)
    metrics['fixed_parameter_ablations'] = ablations
    exported, projection_errors, calibration = export_config(args.export, config, q, notes, flags['two_stage'], flags.get('deployable', False))
    metrics['export_gain_calibration'] = calibration
    exported_audio = []
    for i, note in enumerate(notes):
        matched_pitch = 69+12*np.log2(note['frequency']/440)
        y = engine_render([[0, 0, note['string']-1, 0, matched_pitch, 90, 0, 0, 0, 0, 0]], args.seconds, config=exported)[0]
        y *= .05/max(np.sqrt(np.mean(y**2)), 1e-10)
        exported_audio.append(y)
        sf.write(out/f'note-{i+1}-recording.wav', target[i], 48000, subtype='FLOAT')
        sf.write(out/f'note-{i+1}-current.wav', baseline[i], 48000, subtype='FLOAT')
        sf.write(out/f'note-{i+1}-fitted.wav', fitted[i], 48000, subtype='FLOAT')
        sf.write(out/f'note-{i+1}-export.wav', y, 48000, subtype='FLOAT')
    metrics['export'] = score(np.array(exported_audio), prepared)
    if any(flags.get(k) for k in ('room', 'mic', 'background', 'codec')):
        settings = {k: np.asarray(v).tolist() for k,v in chain.physical(p).items()}
        settings.update(stage_flags={k: flags.get(k, False) for k in ('room','mic','background','codec')},
                        latents={k: np.asarray(v).tolist() for k,v in p.items() if k.startswith('chain_')},
                        mic_response_hz=chain.MIC_GAUGE_HZ.tolist(),
                        mic_response_db=np.asarray(chain.mic_response(p, chain.MIC_GAUGE_HZ)).tolist(),
                        mic_gauge='Mean realized dB at 256 log-spaced frequencies, 300–22050 Hz',
                        background_hz=chain.BACKGROUND_HZ.tolist(), background_psd=notes[0]['background_psd'],
                        note_background_scales=[n['background_scale'] for n in notes],
                        source='Constrained recording nuisance, not an identified physical room or microphone')
        (out/'recording-chain.json').write_text(json.dumps(settings, indent=2)+'\n')
        through = chain.apply(jnp.array(exported_audio), p, notes, **settings['stage_flags'])
        metrics['export_through_chain'] = score(np.array(through), prepared)
        dry_flags = dict(flags, room=False, mic=False, background=False, codec=False)
        metrics['dry_twin'] = score(np.array(twin.render(p, base, config, notes, target.shape[-1], **dry_flags)), prepared)
        metrics['recording_chain'] = settings
    if args.metallic_weight:
        metallic = loss.metallic_prepare(jnp.array(target), notes)
        metrics['metallic_envelope'] = {name: float(loss.metallic_term(jnp.array(y), metallic))
                                        for name, y in [('fitted', fitted), ('export', np.array(exported_audio))]}
    metrics['export_log_rate_rmse'] = projection_errors
    deltas = {}
    for table, values in exported.items():
        for key, value in values.items():
            if key in config[table] and np.shape(value) == np.shape(config[table][key]):
                original = np.asarray(config[table][key])
                fitted_value = np.asarray(value)
                deltas[f'{table}.{key}'] = dict(current=original.tolist(), exported=fitted_value.tolist(), difference=(fitted_value-original).tolist())
    (out/'parameter-deltas.json').write_text(json.dumps(deltas, indent=2)+'\n')
    metrics['per_note'] = []
    for i, note in enumerate(notes):
        local = loss.prepare(jnp.array(target[i:i+1]), [note])
        metrics['per_note'].append(dict(pitch=note['pitch'], current=score(baseline[i:i+1], local), fitted=score(fitted[i:i+1], local),
                                        attack_centroid_hz=dict(recording=attack_centroid(target[i]), current=attack_centroid(baseline[i]), fitted=attack_centroid(fitted[i]))))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True, layout='constrained')
        transforms = [stft(y, 48000, nperseg=1024, noverlap=768) for y in (target[i], baseline[i], fitted[i])]
        peak = max(abs(z).max() for _, _, z in transforms)
        for ax, (frequencies, times, z), title in zip(axes, transforms, ['Recording', 'Current numba model', 'Fitted JAX twin']):
            image = ax.pcolormesh(times, frequencies, 20*np.log10(abs(z)/peak+1e-8), vmin=-70, vmax=0, shading='auto', cmap='magma')
            ax.set(title=title, xlabel='Time from aligned onset (s)', ylim=(80, 10000), yscale='log')
        axes[0].set_ylabel('Frequency (Hz)')
        fig.colorbar(image, ax=axes, label='dB relative to shared peak')
        fig.suptitle(f'MIDI {note["pitch"]} — {note["frequency"]:.2f} Hz; string {note["string"]} assumed')
        fig.savefig(out/f'note-{i+1}-spectrogram.png', dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot(history)
    ax.set(xlabel='Adam step (last point evaluated after update)', ylabel='Regularized objective', title='Shared-instrument fit')
    fig.savefig(out/'loss.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    f = np.geomspace(80, 24000, 300)
    for s in range(5):
        f0 = 440*2**((config['strings']['open_midi'][s]-69)/12)
        mix = config['strings']['loss_mix'][s]
        original = np.log(1000)/config['strings']['t60'][s]-f0*np.log(abs(1-mix+mix*np.exp(-2j*np.pi*f/48000)))
        axes[0].plot(f, original, '--', alpha=.4, color=f'C{s}')
        axes[0].plot(twin.KNOTS, q['rates'][s], 'o-', color=f'C{s}', label=f'String {s+1}')
    axes[0].set(xscale='log', yscale='log', xlabel='Frequency (Hz)', ylabel='Amplitude decay rate (1/s)', title='Solid: learned; dashed: configured loop')
    axes[0].legend(fontsize=8)
    fullf = np.array(config['head']['ratios'])*config['head']['fundamental_hz']
    axes[1].scatter(fullf, config['head']['weights'], s=5, alpha=.25, label='Full configured head')
    axes[1].scatter(base[0], base[2], s=20, facecolors='none', edgecolors='C1', label='Initial selected bank + pot')
    mode_plot = axes[1].scatter(q['frequency'], q['gain'], s=25, c=q['t60'], cmap='viridis', label='Fitted')
    fig.colorbar(mode_plot, ax=axes[1], label='Fitted modal T60 (s)')
    axes[1].set(xlabel='Mode frequency (Hz)', ylabel='Fundamental-relative radiated gain', yscale='log', title='Body modes; no forced pruning')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'instrument.png', dpi=140); plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    active = int(np.sum(q['gain'] > .1*np.array(base[2])))
    current = metrics['current']['data_loss']; fit = metrics['fitted']['data_loss']
    lines = ['# Banjo recording fit', '',
             f'The shared JAX twin reduces aligned data loss from **{current:.4f} to {fit:.4f}** ({100*(1-fit/current):.1f}%); the exported numba projection scores **{metrics["export"]["data_loss"]:.4f}**.',
             'These are recording-matching measurements, not listening judgments or a validated realism score.', '',
             '## Data and identification limits', '',
             f'{len(notes)} isolated Philharmonia notes (eight forte, G4 piano) span MIDI {min(n["pitch"] for n in notes)}–{max(n["pitch"] for n in notes)} at 48 kHz, with the first {args.seconds:g} seconds fitted.',
             'The source does not identify resonator construction, a five-string instrument, picks or strings.',
             'String labels are assumed model assignments; G4 is assigned to string 5, so its B and loss curve are not verified fifth-string measurements.',
             'The recordings are MP3s: codec artifacts, microphone response and room acoustics can be absorbed into fitted body modes and attack noise.',
             'Read [SOURCES.md](SOURCES.md) for URLs, terms and rejected alternatives.',
             'Onsets retain 5 ms before the first 1-ms RMS frame above 15% of peak; targets and current-model comparisons are normalized to 0.05 RMS once.',
             'Reference tuning uses the fundamental when audible, or the two lowest reliable partials when it is absent; it is not forced to equal temperament.',
             'Current and exported numba comparisons are rendered at the measured recording pitch, so tuning correction cannot masquerade as timbre improvement.', '',
             '## Attack and decay evidence', '',
             '| MIDI | Assigned string | Recording tuning (cents) | Glide (cents) | Glide time (ms) | Noise level¹ | Pick width (ms) |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for i, n in enumerate(notes):
        lines.append(f'| {n["pitch"]} | {n["string"]} | {n["tuning_cents"]:.2f} | {q["glide"][i]:.2f} | {q["glide_time"][i]*1000:.2f} | {q["noise"][i]:.4f} | {q["width"][i]*1000:.3f} |')
    lines += ['', '¹ Noise level multiplies a unit-RMS colored burst by 0.001 before note gain and the fixed initial loudness scale.',
              'Glide uses a first-order integrated frequency shift, then differentiable linear interpolation; ±80 cents bounds keep the approximation small.',
              'Fixed-parameter ablations remove a component without allowing the remaining parameters to compensate.',
              'They measure sensitivity, not evidence from separately optimized model selection.', '',
              '| Component removed | Data loss | Difference from full |', '| --- | --- | --- |']
    for name, value in ablations.items():
        lines.append(f'| {name} | {value["data_loss"]:.4f} | {value["data_loss"]-fit:+.4f} |')
    lines += ['', f'**Mode sparsity:** {active}/{len(base[0])} modes retain more than 10% of their initial gain.',
              'The reduced bank starts from the strongest integrated-energy modes of the configured Bessel membrane plus the pot, not the first 39 low modes.',
              'Reducing 950 configured head modes to a 39-mode candidate bank is a design choice, not a fitted finding.',
              'An L1 gain penalty is applied, but positive exponential gains never become exactly zero.',
              '**Two-stage decay:** '+('enabled; its fixed-parameter ablation is above.' if flags['two_stage'] else 'not enabled in this fit; separately refitted controls, when present, are reported below.'), '',
              '## Instrument parameters and export', '',
              'The body is a series filter, with no driving-point feedback or sympathetic strings.',
              'Its modes use causal real biquads and positive gains; modal signs and complex gains are not learned.',
              f'{int(np.sum(q["t60"]>=.999))}/{len(q["t60"])} fitted body modes reach the 1-second damping cap; body/string tails and recording acoustics are not independently identified.',
              'Learned string decay is positive and monotone via cumulative positive knot increments and linear interpolation in log frequency.',
              'The unrestricted twin uses additional B on top of the configured allpass; the deployable twin instead fits three allpass stages.',
              f'The FFT period is {metrics["fft_samples"]/48000:.3f} s; minimum decay rate {twin.MIN_RATE:.3f}/s bounds the periodic tail below −100 dB.',
              'Damping, mode shifts, attack parameters and learning-rate bounds are defined in `twin.py`.', '',
              '| String | Current T60 (s) | Target T60 at 1 kHz (s) | FIR partial log-rate RMSE |',
              '| --- | --- | --- | --- |']
    for s in range(5):
        rate = np.interp(np.log(1000), np.log(twin.KNOTS), q['rates'][s])
        lines.append(f'| {s+1} | {config["strings"]["t60"][s]:.3f} | {np.log(1000)/rate:.3f} | {projection_errors[s]:.3f} |')
    lines += ['', 'The TOML uses the engine’s existing explicit modal-array schema; no default engine path is changed.',
              'The engine designs a contractive minimum-phase FIR from the learned T60 curve at each pitch; stiffness uses three allpass stages.',
              'The deployable export shares excitation by string and disables bridge coupling/feedback to reflect the series-body approximation.',
              'The exported preset enables an exponential pick pulse; global excitation medians are fallbacks for profiles without per-string arrays.',
              f'The export stage retains output gain {calibration["output_gain_after"]:.3f}; simultaneous fortissimo probe peak is {calibration["probe_peak_after"]:.3f}.',
              'The delivery stage calibrates instrument gain to default étude LUFS and verifies decoded MP3 loudness and true peak.',
              'Pitch glide, colored pick noise and per-note excitation are not represented in this TOML; the deployable second polarization uses per-string mean mixes.',
              'The separately measured export score above quantifies this loss of fidelity; do not assume the export sounds like the fitted WAVs.',
              'Every learned physical parameter is in `metrics.json`; mode-by-mode shifts are in `body-modes.json`, and current/exported parameter differences are in `parameter-deltas.json`.', '',
              '## Attack brightness', '',
              'Magnitude-weighted spectral centroid uses the first 50 ms, a Hann window and the 80 Hz–10 kHz band; it is not the survey’s estimator unless those settings match.', '',
              '| MIDI | Recording centroid (Hz) | Current centroid (Hz) | Fitted centroid (Hz) |',
              '| --- | --- | --- | --- |']
    for row in metrics['per_note']:
        cents = row['attack_centroid_hz']
        lines.append(f'| {row["pitch"]} | {cents["recording"]:.0f} | {cents["current"]:.0f} | {cents["fitted"]:.0f} |')
    lines += ['', '## Figures and reproduction', '', '![Objective](loss.png)', '', '![Decay curves and body modes](instrument.png)', '']
    for i, n in enumerate(notes):
        lines += [f'![MIDI {n["pitch"]}: recording/current/fitted](note-{i+1}-spectrogram.png)', '']
    lines += ['```sh', 'uv run scripts/models/banjo_fit/fetch.py',
              f'uv run scripts/models/banjo_fit/fit.py --notes {args.notes} --out {args.out} --steps {args.steps} --modes {len(base[0])} --initial-b {args.initial_b:g} --sparsity {args.sparsity:g} --learning-rate {args.learning_rate:g} --export {args.export}'+(' --two-stage' if flags['two_stage'] else '')+(' --no-glide' if not flags['glide'] else '')+(' --no-noise' if not flags['noise'] else '')+(' --deployable' if flags.get('deployable') else '')+(f' --metallic-weight {args.metallic_weight}' if args.metallic_weight else '')+(' --freeze-body' if args.freeze_body else '')+(' --fit-body-gains' if args.fit_body_gains else '')+(f' --resume {args.resume}' if args.resume else ''),
              'uv run tests/test_banjo_fit.py', '```', '',
              'Fit settings, per-note scores and fixed-parameter ablations are in `metrics.json`.',
              'WAVs are unmastered, loudness-aligned comparisons; no auditory evaluation has been performed.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'body-modes.json').write_text(json.dumps([dict(initial_hz=float(f), fitted_hz=float(nf), initial_t60=float(t), fitted_t60=float(nt), initial_gain=float(g), fitted_gain=float(ng)) for f,nf,t,nt,g,ng in zip(base[0],q['frequency'],base[1],q['t60'],base[2],q['gain'])], indent=2))
    print(out/'REPORT.md', flush=True)
