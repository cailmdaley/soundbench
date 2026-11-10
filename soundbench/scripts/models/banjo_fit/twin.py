"""Differentiable force-driven banjo; bridge/body coupling is a series filter.

The exact discrete loop is the calibration reference. Learned loss replaces its
magnitude, retaining its calibrated phase. B adds a stiff-string phase increment
on top of the engine's dispersion, so B=0 means the configured allpass stiffness.
The FFT period is chosen to suppress a worst-case three-second T60 by 100 dB.
"""
from __future__ import annotations
import math
import numpy as np
import jax
import jax.numpy as jnp
from scipy.signal import freqz
from soundbench.scripts.models.banjo import _coefficients, modal_coefficients

# Narrow waveguide/body poles need double precision for engine parity.
jax.config.update('jax_enable_x64', True)

SR = 48000
KNOTS = np.array([80., 300., 600., 1000., 1500., 2000., 3000., 5000., 8000., 24000.])
MIN_RATE = math.log(1000) / 3


def fft_size(seconds):
    return 2 ** math.ceil(math.log2(max(seconds + .5, 5.) * SR))


def exact_loop(frequency, pitch, string, config):
    """Numba's integer delay, centered Thiran, dispersion and two-tap loss."""
    s = config['strings']
    integer, a, gain = _coefficients(float(pitch), SR, s['loss_mix'][string],
                                    s['dispersion'][string], s['t60'][string])
    z = jnp.exp(-2j * jnp.pi * frequency / SR)
    d = s['dispersion'][string]
    loop = gain * z**integer * (a + z)/(1 + a*z) * (d + z)/(1 + d*z)
    return loop * (1 - s['loss_mix'][string] + s['loss_mix'][string]*z)


def string_response(frequency, pitch, string, config):
    return 1 / (1 - exact_loop(frequency, pitch, string, config))


def initial_parameters(config, notes, modes=40):
    """Select 39 strongest radiating head modes across the bank plus the pot."""
    c, r2, b, weights = modal_coefficients(config)
    h = config['head']
    frequencies = np.array(h['ratios']) * h['fundamental_hz']
    times = np.array(h['t60_seconds'])
    # Integrated modal energy, not frequency order, chooses the reduced bank.
    energy = weights[:-1]**2 * times
    chosen = np.sort(np.argsort(energy)[-(modes-1):])
    chosen = np.unique(np.r_[0, chosen])
    if len(chosen) > modes-1:
        chosen = np.delete(chosen, np.argmin(energy[chosen[1:]])+1)
    fs = np.r_[frequencies[chosen], config['pot']['frequency_hz']]
    ts = np.r_[times[chosen], config['pot']['t60_seconds']]
    fundamental_weight = h['weights'][np.argmin(h['ratios'])]
    ws = np.r_[np.array(h['weights'])[chosen]/fundamental_weight, config['pot']['weight']]
    rates = []
    for s in range(5):
        f0 = 440*2**((config['strings']['open_midi'][s]-69)/12)
        mix = config['strings']['loss_mix'][s]
        z = np.exp(-2j*np.pi*KNOTS/SR)
        rate = np.log(1000)/config['strings']['t60'][s] - f0*np.log(abs(1-mix+mix*z))
        increments = np.diff(np.r_[MIN_RATE, rate])
        rates.append(np.log(np.maximum(increments, .01)))
    count = len(notes)
    return dict(loss=jnp.array(rates), b=jnp.full(5, -6.),
                mode_frequency=jnp.zeros(len(fs)), mode_time=jnp.zeros(len(fs)),
                mode_gain=jnp.array(np.log(np.maximum(ws, 1e-5))),
                width=jnp.zeros(count), corner=jnp.zeros(count), position=jnp.zeros(count),
                level=jnp.zeros(count), detune=jnp.zeros(count), onset=jnp.zeros(count),
                glide=jnp.zeros(count), glide_time=jnp.zeros(count),
                noise=jnp.full(count, -5.), noise_time=jnp.zeros(count), tilt=jnp.zeros(count),
                stages=jnp.full((5, 3), -1.), second_mix=jnp.full(count, -4.), second_rate=jnp.zeros(5), second_detune=jnp.zeros(5)), (fs, ts, ws)


def physical(p, base, config):
    fs, ts, ws = base
    return dict(stages=-.98*jax.nn.sigmoid(p['stages']), rates=MIN_RATE+jnp.cumsum(jnp.exp(jnp.clip(p['loss'], -12, 9)), axis=-1),
                B=.02 * jax.nn.sigmoid(p['b']),
                frequency=jnp.array(fs)*jnp.exp(.25*jnp.tanh(p['mode_frequency'])),
                t60=jnp.clip(jnp.array(ts)*jnp.exp(p['mode_time']), .015, 1.),
                gain=jnp.exp(jnp.clip(p['mode_gain'], -15, 6)),
                width=config['excitation']['pick_ms']/1000*jnp.exp(jnp.clip(p['width'], -2, 2)),
                corner=config['excitation']['lowpass_hz']*jnp.exp(jnp.clip(p['corner'], -3, 1)),
                position=.02+.46*jax.nn.sigmoid(p['position'] + math.log((config['excitation']['pluck_position']-.02)/(.48-config['excitation']['pluck_position']))),
                level=jnp.exp(jnp.clip(p['level'], -5, 5)), detune=50*jnp.tanh(p['detune']),
                onset=.003*jnp.tanh(p['onset']), glide=80*jnp.tanh(p['glide']),
                glide_time=.015*jnp.exp(jnp.clip(p['glide_time'], -2, 2)),
                noise=jnp.exp(jnp.clip(p['noise'], -15, 2)),
                noise_time=.005*jnp.exp(jnp.clip(p['noise_time'], -2, 2)),
                tilt=2*jnp.tanh(p['tilt']), second_mix=.5*jax.nn.sigmoid(p['second_mix']),
                second_rate=jnp.exp(jnp.clip(p['second_rate'], -3, 3)),
                second_detune=8*jnp.tanh(p['second_detune']))


def canonical_excitation(p, base, config, notes):
    """Represent shared physical excitation with identical per-string latents.

    Averaging clipped per-note physical values leaves saturated latent values
    unable to move. The canonical representation preserves the rendered sound
    while putting the shared excitation inside its trainable bounds.
    """
    parameters = dict(p)
    q = physical(p, base, config)
    position_offset = math.log((config['excitation']['pluck_position']-.02)/(.48-config['excitation']['pluck_position']))
    for key in ('width', 'corner', 'position', 'second_mix'):
        values = q[key]
        shared = jnp.array([jnp.mean(values[jnp.array([k for k,n in enumerate(notes) if n['string']==note['string']])]) for note in notes])
        if key == 'width':
            latent = jnp.log(shared/(config['excitation']['pick_ms']/1000))
        elif key == 'corner':
            latent = jnp.log(shared/config['excitation']['lowpass_hz'])
        else:
            fraction = (shared-.02)/.46 if key == 'position' else 2*shared
            fraction = jnp.clip(fraction, 1e-12, 1-1e-12)
            latent = jnp.log(fraction/(1-fraction))-(position_offset if key=='position' else 0.)
        parameters[key] = latent
    return parameters


def reseed_bounded_latents(p, base, margin=.25):
    """Move saturated/clipped coordinates just inside their active bounds.

    The transform preserves every physical value that is already at least
    ``margin`` latent units from a clipping boundary. Values outside that
    interior are moved to the nearest interior point, retaining a usable
    derivative on the next optimization step.
    """
    if not 0 < margin < 1:
        raise ValueError('margin must be between zero and one')
    result = {key: jnp.array(value) for key, value in p.items()}

    def inside(key, low, high):
        if key in result:
            result[key] = jnp.clip(result[key], low + margin, high - margin)

    inside('loss', -12., 9.)
    inside('mode_gain', -15., 6.)
    inside('width', -2., 2.)
    inside('corner', -3., 1.)
    inside('glide_time', -2., 2.)
    inside('noise', -15., 2.)
    inside('noise_time', -2., 2.)
    inside('second_rate', -3., 3.)
    # mode_time is clipped in physical T60 space, whose latent bounds depend
    # on each initialized modal decay time.
    if 'mode_time' in result:
        base_t60 = jnp.asarray(base[1])
        low = jnp.log(.015 / base_t60) + margin
        high = jnp.log(1. / base_t60) - margin
        result['mode_time'] = jnp.minimum(jnp.maximum(result['mode_time'], low), high)
    # These coordinates pass through sigmoid/tanh and have no hard clip, but
    # values beyond ±4 make their gradients too small to move reliably.
    inside('b', -4., 4.)
    inside('mode_frequency', -4., 4.)
    inside('stages', -4., 4.)
    for key in ('position', 'second_mix', 'detune', 'onset', 'glide', 'tilt', 'second_detune'):
        if key in result:
            result[key] = jnp.clip(result[key], -4., 4.)
    inside('level', -5., 5.)
    return result


def decay_rate(frequency, rates, multiplier=1.):
    """Monotone loop decay, with the FFT tail floor for either polarization."""
    interpolated = jnp.interp(jnp.log(jnp.maximum(frequency, 1)), jnp.log(jnp.array(KNOTS)), rates)
    return jnp.maximum(interpolated*multiplier, MIN_RATE)


def body_response(f, q, config):
    """Real causal biquads with nonnegative radiated gains (fixed positive sign)."""
    z = jnp.exp(-2j*jnp.pi*f/SR)
    # scan avoids allocating frequency-grid × mode-bank intermediate arrays.
    def add(total, mode):
        hz, t60, weight = mode
        r = jnp.exp(-jnp.log(1000)/(SR*t60))
        angle = 2*jnp.pi*hz/SR
        b = .5*(t60/.12)*(1-r)*jnp.abs(1-r*jnp.exp(-2j*angle))
        return total + weight*b/(1-2*r*jnp.cos(angle)*z+r*r*z*z), None
    response, _ = jax.lax.scan(add, jnp.zeros_like(z), (q['frequency'], q['t60'], q['gain']))
    return config['pickups']['string'] + config['pickups']['head']*.03931491719806966*response


def fir_loss(rates, f0):
    """Differentiable homomorphic minimum-phase FIR, matching string_loss.design_loss."""
    grid = jnp.linspace(0., 24000., 257)
    target = jnp.exp(-2*decay_rate(grid, rates)/f0)
    spectrum = target*jnp.exp(-2j*jnp.pi*grid*64/SR)
    linear = jnp.fft.irfft(spectrum, 512)[:129]*jnp.hamming(129)
    spectrum = jnp.fft.fft(linear, 32768)
    magnitude = jnp.sqrt(spectrum.real**2+spectrum.imag**2+1e-18)
    log = .5*jnp.log(magnitude+1e-7*jnp.min(magnitude))
    cepstrum = jnp.fft.ifft(log).real
    causal = jnp.concatenate((jnp.ones(1), jnp.full(16383, 2.), jnp.ones(1), jnp.zeros(16383)))
    h = jnp.fft.ifft(jnp.exp(jnp.fft.fft(cepstrum*causal))).real[:65]
    spectrum = jnp.fft.fft(h, 32768)
    response = jnp.sqrt(spectrum.real**2+spectrum.imag**2+1e-18)
    bound = jnp.max(response)+jnp.pi/32768*jnp.sum(jnp.arange(65)*jnp.abs(h))
    return h*jnp.minimum(1., (1-1e-7)/bound)


def discrete_loop(f, pitch, rates, dispersion, stages):
    """Exact FIR, dispersion cascade and fundamental-compensated Thiran loop."""
    low = jnp.floor(pitch)
    fraction = pitch-low
    h = (1-fraction)*fir_loss(rates, 440*2**((low-69)/12)) + fraction*fir_loss(rates, 440*2**((low+1-69)/12))
    f0 = 440*2**((pitch-69)/12)
    w0 = 2*jnp.pi*f0/SR
    phase = -jnp.angle(jnp.sum(h*jnp.exp(-1j*w0*jnp.arange(65))))
    ds = jnp.concatenate((jnp.array([dispersion]), stages))
    phase += jnp.sum(2*jnp.arctan((1-ds)/(1+ds)*jnp.tan(w0/2)))
    delay = SR/f0-phase/w0
    integer = jnp.floor(delay-.5)
    t, u = jnp.tan(w0/2), jnp.tan(w0*(delay-integer)/2)
    a = (t-u)/(t+u)
    z = jnp.exp(-2j*jnp.pi*f/SR)
    loop = z**integer*(a+z)/(1+a*z)*jnp.fft.rfft(h, (len(f)-1)*2)
    for d in ds:
        loop *= (d+z)/(1+d*z)
    return loop


def deployable_render(q, config, notes, samples):
    """Sequential mapped exact loops keep the nine-note gradient graph bounded."""
    nfft = fft_size(samples/SR)
    f = jnp.fft.rfftfreq(nfft, 1/SR)
    z = jnp.exp(-2j*jnp.pi*f/SR)
    body = body_response(f, q, config)
    strings = jnp.array([n['string']-1 for n in notes])
    frequencies = jnp.array([n['frequency'] for n in notes])
    dispersion = jnp.array(config['strings']['dispersion'])
    def one(values):
        i, s, f0 = values
        pitch = 69+12*jnp.log2(f0/440)
        loop = discrete_loop(f, pitch, q['rates'][s], dispersion[s], q['stages'][s])
        string = 1/(1-loop)
        if q.get('two_stage', False):
            second = discrete_loop(f, pitch+q['second_detune'][s]/100,
                                   jnp.maximum(q['rates'][s]*q['second_rate'][s], MIN_RATE), dispersion[s], q['stages'][s])
            string = (1-q['second_mix'][i])*string+q['second_mix'][i]/(1-second)
        alpha = 1-jnp.exp(-2*jnp.pi*jnp.minimum(q['corner'][i], 20000)/SR)
        pole = jnp.exp(-2/(SR*q['width'][i]))
        pulse = (1-pole)/(1-pole*z)*alpha/(1-(1-alpha)*z)
        travel = jnp.maximum(1., q['position'][i]*SR/f0)
        comb = jnp.floor(travel)
        fraction = travel-comb
        if config['excitation'].get('fractional_pluck', False):
            comb_delay = (1-fraction)*z**comb+fraction*z**(comb+1)
        else:
            comb_delay = z**comb
        spectrum = pulse*(1-comb_delay)*string
        if q.get('two_stage', False):
            # Each polarization computes pluck travel time at its own pitch.
            second_f0 = f0*2**(q['second_detune'][s]/1200)
            second_travel = jnp.maximum(1., q['position'][i]*SR/second_f0)
            second_comb = jnp.floor(second_travel)
            second_fraction = second_travel-second_comb
            if config['excitation'].get('fractional_pluck', False):
                second_delay = (1-second_fraction)*z**second_comb+second_fraction*z**(second_comb+1)
            else:
                second_delay = z**second_comb
            spectrum += pulse*q['second_mix'][i]*(comb_delay-second_delay)/(1-second)
        raw = jnp.fft.irfft(spectrum*body, nfft)[:samples]
        return raw*.05/jnp.sqrt(jnp.mean(raw**2)+1e-20)
    return jax.lax.map(jax.checkpoint(one), (jnp.arange(len(notes)), strings, frequencies))


def render(p, base, config, notes, samples, *, glide=True, noise=True, two_stage=False, deployable=False,
           room=False, mic=False, background=False, codec=False):
    nfft = fft_size(samples/SR)
    f = jnp.fft.rfftfreq(nfft, 1/SR)
    q = physical(p, base, config)
    if deployable:
        glide = noise = False
        q['onset'] = jnp.zeros_like(q['onset'])
        for key in ('width', 'corner', 'position', 'second_mix'):
            values = q[key]
            q[key] = jnp.array([jnp.mean(values[jnp.array([k for k, n in enumerate(notes) if n['string'] == note['string']])]) for note in notes])
    if deployable:
        q['two_stage'] = two_stage
        dry = deployable_render(q, config, notes, samples)
        if room or mic or background or codec:
            from soundbench.scripts.models.banjo_fit import chain
            return chain.apply(dry, p, notes, room=room, mic=mic, background=background, codec=codec)
        return dry
    body = body_response(f, q, config)
    time = jnp.arange(samples)/SR
    outputs = []
    for i, note in enumerate(notes):
        s, pitch = note['string']-1, note['pitch']
        f0 = note['frequency'] * 2**(q['detune'][i]/1200)
        nominal = 440*2**((pitch-69)/12)
        shifted = f*nominal/f0
        baseline = exact_loop(shifted, pitch, s, config)
        rate = decay_rate(f, q['rates'][s])
        ratio = f/f0
        B = q['B'][s]
        stiff_order = ratio*jnp.sqrt(2*(1+B)/(1+jnp.sqrt(1+4*B*(ratio**2)*(1+B))))
        phase = baseline / jnp.maximum(jnp.abs(baseline), 1e-8) * jnp.exp(-2j*jnp.pi*(stiff_order-ratio))
        loop = jnp.exp(-rate/f0)*phase
        string = 1/(1-loop)
        if two_stage:
            second_decay = decay_rate(f, q['rates'][s], q['second_rate'][s])
            second = jnp.exp(-second_decay/f0)*phase*jnp.exp(2j*jnp.pi*ratio*q['second_detune'][s]/1200*jnp.log(2))
            string = (1-q['second_mix'][i])*string + q['second_mix'][i]/(1-second)
        if deployable:
            engine_pitch = 69+12*jnp.log2(note['frequency']/440)
            loop = discrete_loop(f, engine_pitch, q['rates'][s], config['strings']['dispersion'][s], q['stages'][s])
            string = 1/(1-loop)
            if two_stage:
                second_pitch = engine_pitch+q['second_detune'][s]/100
                second = discrete_loop(f, second_pitch, jnp.maximum(q['rates'][s]*q['second_rate'][s], MIN_RATE), config['strings']['dispersion'][s], q['stages'][s])
                string = (1-q['second_mix'][i])*string + q['second_mix'][i]/(1-second)
        # Exponential pick pulse is continuous in width; area is one.
        alpha = 1-jnp.exp(-2*jnp.pi*jnp.minimum(q['corner'][i], 20000)/SR)
        z = jnp.exp(-2j*jnp.pi*f/SR)
        pulse = 1/(1+2j*jnp.pi*f*q['width'][i]/2) * alpha/(1-(1-alpha)*z)
        excitation = pulse * (1-jnp.exp(-2j*jnp.pi*f*q['position'][i]/f0))
        if deployable:
            pole = jnp.exp(-2/(SR*q['width'][i]))
            pulse = (1-pole)/(1-pole*z)*alpha/(1-(1-alpha)*z)
            nominal_f0 = note['frequency']
            comb = jnp.maximum(1, jnp.floor(q['position'][i]*SR/nominal_f0))
            excitation = pulse*(1-z**comb)
        raw = jnp.fft.irfft(excitation*string*body, nfft)[:samples]
        # Integrated instantaneous frequency gives a differentiable sample clock.
        clock = time
        if glide:
            delta = q['glide'][i]*jnp.log(2)/1200
            clock = clock + delta*q['glide_time'][i]*(1-jnp.exp(-time/q['glide_time'][i]))
        raw = jnp.interp(clock-q['onset'][i], time, raw, left=0., right=0.)
        if noise:
            rng = jax.random.PRNGKey(1729+i)
            white = jax.random.normal(rng, (samples,))
            nf = jnp.fft.rfftfreq(samples, 1/SR)
            color = ((nf+200)/1000)**q['tilt'][i]
            burst = jnp.fft.irfft(jnp.fft.rfft(white)*color, samples)
            burst = burst/jnp.sqrt(jnp.mean(burst**2)+1e-8)
            age = time-q['onset'][i]
            raw = raw + .001*q['noise'][i]*burst*jnp.exp(-jnp.maximum(age, 0)/q['noise_time'][i])*(age>=0)
        # Fixed per-note reference scale is supplied by the fitter, not normalized here.
        if deployable:
            raw = raw*.05/jnp.sqrt(jnp.mean(raw**2)+1e-20)
            outputs.append(raw)
        else:
            outputs.append(raw*q['level'][i]*note.get('scale', 1.))
    dry = jnp.stack(outputs)
    if room or mic or background or codec:
        from soundbench.scripts.models.banjo_fit import chain
        return chain.apply(dry, p, notes, room=room, mic=mic, background=background, codec=codec)
    return dry
