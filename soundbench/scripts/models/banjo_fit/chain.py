"""Shared, bounded recording stages; none are instrument export parameters.

Room uses a fixed direct path, four early taps and a seeded two-band late IR.
Mic EQ uses resolvable controls and zero mean realized log gain (gain gauge).
Background PSD is measured before onset, not inferred from the pluck tail.
Codec is a signal-dependent shaped-noise proxy, not an MP3 implementation.
All convolutions are zero padded; no tail wraps into the attack.
"""
import numpy as np
import jax
import jax.numpy as jnp

SR = 48000
EQ_HZ = np.array([300., 600., 1000., 2000., 4000., 9000., 22050.])
MIC_GAUGE_HZ = np.geomspace(300., 22050., 256)
BACKGROUND_HZ = np.geomspace(10., 24000., 24)
IR_SAMPLES = 9600


def initial_parameters(*, zero_start=False):
    """Return the historical chain seed or an exact identity-chain seed.

    The opt-in seed marks itself with an empty array leaf so the marker survives
    optimizer pytrees and snapshots without becoming a fitted boolean. Its
    signed tanh amplitudes let room, measured-background noise and codec noise
    leave zero with finite gradients.
    """
    if not zero_start:
        return dict(chain_room=jnp.full(5, -3.), chain_t60=jnp.zeros(2),
                    chain_mic=jnp.zeros(7), chain_background=jnp.array(0.),
                    chain_codec=jnp.array(-2.))
    return dict(chain_room=jnp.zeros(5), chain_t60=jnp.zeros(2),
                chain_mic=jnp.zeros(7), chain_background=jnp.array(0.),
                chain_codec=jnp.array(0.), chain_codec_rolloff=jnp.array(0.),
                chain_zero_start=jnp.zeros((0,), dtype=jnp.float32))


def _zero_start(p):
    return 'chain_zero_start' in p


def mic_filter(p):
    """Causal 129-tap FIR, gauge-fixed on its realized log-frequency response.

    Controls below 300 Hz are tied to the 300-Hz control rather than claiming
    independently realizable bass anchors. The gauge is the arithmetic mean dB
    at 256 logarithmically spaced frequencies from 300 to 22050 Hz.
    """
    eq = 4*jnp.tanh(p['chain_mic'])
    eq -= jnp.mean(eq)
    grid = jnp.fft.rfftfreq(512, 1/SR)
    db = jnp.interp(jnp.log(jnp.maximum(grid, 1)), jnp.log(jnp.array(EQ_HZ)), eq)
    cep = jnp.fft.irfft(db*np.log(10)/20, 512)
    causal = jnp.concatenate((jnp.ones(1), jnp.full(255, 2.), jnp.ones(1), jnp.zeros(255)))
    h = jnp.fft.ifft(jnp.exp(jnp.fft.fft(cep*causal))).real[:129]
    response = jnp.exp(-2j*jnp.pi*jnp.array(MIC_GAUGE_HZ)[:,None]*jnp.arange(129)/SR) @ h
    mean_log = jnp.mean(jnp.log(jnp.maximum(jnp.abs(response), 1e-12)))
    return h*jnp.exp(-mean_log)


def mic_response(p, frequencies=EQ_HZ):
    h = mic_filter(p)
    response = jnp.exp(-2j*jnp.pi*jnp.array(frequencies)[:,None]*jnp.arange(129)/SR) @ h
    return 20*jnp.log10(jnp.maximum(jnp.abs(response), 1e-12))


def physical(p):
    """Map latents to bounded stage values.

    In the opt-in mode, background_db is relative to the measured PSD's
    amplitude scale (with a finite display floor); background_amplitude is
    the signed multiplier applied to that measured noise realization. The
    codec fraction is signed shaped-noise amplitude, and codec_rolloff blends
    the fixed high-frequency curve from unity at zero to the full curve at
    one. Signed values let nuisance draws leave the exact zero boundary.
    """
    eq = mic_response(p)
    if _zero_start(p):
        background_amplitude = jnp.tanh(p['chain_background'])
        codec_fraction = .01*jnp.tanh(p['chain_codec'])
        codec_rolloff = jnp.tanh(p['chain_codec_rolloff'])
        return dict(room_early=.08*jnp.tanh(p['chain_room'][:4]),
                    room_wet=.15*jnp.tanh(p['chain_room'][4]),
                    room_t60=.04+.26*jax.nn.sigmoid(p['chain_t60']),
                    mic_db=eq,
                    background_db=20*jnp.log10(jnp.maximum(jnp.abs(background_amplitude), 1e-12)),
                    background_amplitude=background_amplitude,
                    codec_fraction=codec_fraction,
                    codec_rolloff=codec_rolloff)
    return dict(room_early=.08*jax.nn.sigmoid(p['chain_room'][:4]),
                room_wet=.15*jax.nn.sigmoid(p['chain_room'][4]),
                room_t60=.04+.26*jax.nn.sigmoid(p['chain_t60']),
                mic_db=eq, background_db=3*jnp.tanh(p['chain_background']),
                codec_fraction=.0001+.0099*jax.nn.sigmoid(p['chain_codec']))


def room_ir(p):
    q = physical(p)
    t = jnp.arange(IR_SAMPLES)/SR
    rng = np.random.default_rng(7163)
    noise = rng.standard_normal(IR_SAMPLES)
    f = np.fft.rfftfreq(IR_SAMPLES, 1/SR)
    low = np.fft.irfft(np.fft.rfft(noise)/(1+(f/1800)**4), IR_SAMPLES)
    high = noise-low
    field = jnp.array(np.stack([low, high]))*jnp.exp(-np.log(1000)*t[None,:]/q['room_t60'][:,None])
    field = jnp.sum(field/jnp.sqrt(jnp.sum(field**2, axis=1, keepdims=True)+1e-20), axis=0)/np.sqrt(2)
    # A late field starts after 12 ms, with a smooth onset and end taper.
    field *= jnp.clip((t-.012)/.004, 0, 1)*jnp.minimum(1., (IR_SAMPLES/SR-t)/.02)
    ir = q['room_wet']*field
    ir = ir.at[0].add(1.)
    return ir.at[jnp.array([331, 619, 1061, 1777])].add(q['room_early'])


def apply(y, p, notes, *, room=False, mic=False, background=False, codec=False, realization=0):
    """Instrument -> room -> mic + constrained background -> codec proxy."""
    if not (room or mic or background or codec):
        return y
    samples = y.shape[-1]
    nfft = 2**int(np.ceil(np.log2(samples+IR_SAMPLES-1)))
    f = jnp.fft.rfftfreq(nfft, 1/SR)
    q = physical(p)
    if room:
        y = jnp.fft.irfft(jnp.fft.rfft(y, nfft)*jnp.fft.rfft(room_ir(p), nfft), nfft)[:,:samples]
    if mic:
        h = mic_filter(p)
        y = jnp.fft.irfft(jnp.fft.rfft(y, nfft)*jnp.fft.rfft(h, nfft), nfft)[:,:samples]
    if background:
        rows = []
        for i, note in enumerate(notes):
            psd = jnp.interp(jnp.log(jnp.maximum(f, 1)), jnp.log(jnp.array(BACKGROUND_HZ)), jnp.log(jnp.array(note['background_psd'])+1e-30))
            white = jnp.array(np.random.default_rng(9929+i+100*realization).standard_normal(nfft))
            noise = jnp.fft.irfft(jnp.fft.rfft(white)*jnp.sqrt(jnp.exp(psd)*SR/2), nfft)[:samples]
            rows.append(noise*note['background_scale'])
        if _zero_start(p):
            y = y+jnp.stack(rows)*q['background_amplitude']
        else:
            y = y+jnp.stack(rows)*10**(q['background_db']/20)
    if codec:
        # Local masking envelope, colored with the signal's smoothed spectrum.
        # Fixed random phases let gradients pass through masking and amplitude.
        spectrum = jnp.fft.rfft(y, nfft)
        power = jnp.abs(spectrum)**2
        smooth = jax.vmap(lambda a: jnp.convolve(a, jnp.ones(65)/65, mode='same'))(power)
        phases = jnp.array(np.exp(2j*np.pi*np.random.default_rng(3109+realization).random(smooth.shape)))
        error = jnp.fft.irfft(jnp.sqrt(smooth+1e-20)*phases, nfft)[:,:samples]
        # 10-ms block RMS interpolates the time-varying codec masking envelope.
        padded = jnp.pad(y, ((0,0),(0,(-samples)%480)))
        env = jnp.sqrt(jnp.mean(padded.reshape(len(notes),-1,480)**2, axis=-1)+1e-12)
        clock = jnp.arange(samples)/480
        envelope = jax.vmap(lambda e: jnp.interp(clock, jnp.arange(e.shape[0]), e))(env)
        envelope /= jnp.sqrt(jnp.mean(y**2, axis=-1, keepdims=True)+1e-12)
        y = y+q['codec_fraction']*error*envelope
        cutoff = 1/jnp.sqrt(1+(f/20500)**32)
        if _zero_start(p):
            cutoff = 1+q['codec_rolloff']*(cutoff-1)
        y = jnp.fft.irfft(jnp.fft.rfft(y, nfft)*cutoff, nfft)[:,:samples]
    return y


def penalty(p, *, room=False, mic=False, background=False, codec=False, **unused):
    q = physical(p)
    if _zero_start(p):
        return ((.01*jnp.mean((q['mic_db']/4)**2) if mic else 0.)
                +(.002*(q['room_wet']/.15)**2 if room else 0.)
                +(.002*q['background_amplitude']**2 if background else 0.)
                +(.002*(q['codec_fraction']/.01)**2 if codec else 0.)
                +(.002*q['codec_rolloff']**2 if codec else 0.))
    return ((.01*jnp.mean((q['mic_db']/4)**2) if mic else 0.)
            +(.002*(q['room_wet']/.15)**2 if room else 0.)
            +(.002*(q['background_db']/3)**2 if background else 0.)
            +(.002*(q['codec_fraction']/.01)**2 if codec else 0.))
