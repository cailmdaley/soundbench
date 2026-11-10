"""Auraloss-style spectral convergence, log magnitude and partial envelopes."""
import jax.numpy as jnp
import numpy as np

WINDOWS = (64, 128, 256, 512, 1024, 2048, 4096)


def magnitude(y, window):
    # Pole synthesis stays float64; float32 STFT workspaces bound fit memory.
    y = y.astype(jnp.float32)
    hop = window//4
    padded = jnp.pad(y, ((0, 0), (window//2, window//2)))
    starts = jnp.arange(0, y.shape[-1], hop)
    frames = padded[:, starts[:, None]+jnp.arange(window)]
    return jnp.sqrt(jnp.abs(jnp.fft.rfft(frames*jnp.hanning(window).astype(jnp.float32), axis=-1))**2+1e-10)


def prepare(target, notes):
    spectra = tuple(magnitude(target, w) for w in WINDOWS)
    # Three-bin windows around each partial, with a separate mask per note.
    frequencies = np.fft.rfftfreq(4096, 1/48000)
    masks = []
    for note in notes:
        partials = note['frequency']*np.arange(1, 17)
        mask = (abs(frequencies[None, :]-partials[:, None]) < 48000/4096*1.5).astype(float)
        mask /= np.maximum(mask.sum(axis=-1, keepdims=True), 1)
        masks.append(mask)
    masks = jnp.array(masks, dtype=jnp.float32)
    envelopes = jnp.einsum('ntf,npf->ntp', spectra[-1], masks)
    # Ignore partials below the recording's meaningful spectral floor.
    weights = (jnp.max(envelopes, axis=1) > .01*jnp.max(envelopes, axis=(1, 2))[:, None]).astype(float)
    return spectra, masks, envelopes, weights


def terms(predicted, prepared):
    target, masks, envelopes, weights = prepared
    spectral = 0.
    for w, reference in zip(WINDOWS, target):
        estimate = magnitude(predicted, w)
        # A fixed noise floor prevents silence from dominating log-magnitude error.
        floor = .001*jnp.max(reference, axis=(1, 2), keepdims=True)+1e-5
        log = jnp.mean(jnp.abs(jnp.log(estimate+floor)-jnp.log(reference+floor)))
        convergence = jnp.mean(jnp.sqrt(jnp.sum((estimate-reference)**2, axis=(1, 2))+1e-8)/
                               (jnp.sqrt(jnp.sum(reference**2, axis=(1, 2)))+1e-8))
        spectral += log+convergence
    predicted_env = jnp.einsum('ntf,npf->ntp', magnitude(predicted, 4096), masks)
    floor = .001*jnp.max(envelopes, axis=1, keepdims=True)+1e-5
    envelope = jnp.sum(jnp.abs(jnp.log(predicted_env+floor)-jnp.log(envelopes+floor))*weights[:, None, :]) / (jnp.sum(weights)*envelopes.shape[1]+1e-8)
    return spectral/len(WINDOWS), envelope


def metallic_prepare(target, notes):
    """Equal-weight resolved upper partials and high bands in the first 300 ms.

    Reference-relative floors retain quiet but audible harmonics without fitting
    silence. Separate attack/tail weights keep short attacks from being diluted
    by the two-second training window.
    """
    frequencies = np.fft.rfftfreq(2048, 1/48000)
    masks = []
    for note in notes:
        centers = note['frequency']*np.arange(1, 26)
        mask = (abs(frequencies[None, :]-centers[:, None]) < max(35., .22*note['frequency'])).astype(float)
        mask /= np.maximum(mask.sum(axis=1, keepdims=True), 1)
        bands = np.array([(frequencies >= lo)&(frequencies < hi) for lo,hi in [(4000,6000),(6000,9000),(9000,12000)]], dtype=float)
        bands /= np.maximum(bands.sum(axis=1, keepdims=True), 1)
        masks.append(np.concatenate((mask, bands)))
    masks = jnp.array(masks)
    spectrum = magnitude(target, 2048)
    envelopes = jnp.einsum('ntf,npf->ntp', spectrum, masks)
    times = np.arange(envelopes.shape[1])*512/48000
    temporal = jnp.array(np.where((times>=.005)&(times<=.030), 4., np.where((times>.030)&(times<=.300), 1., 0.)))
    peak = jnp.max(envelopes* (temporal>0)[None,:,None], axis=1, keepdims=True)
    audible = peak > .0003*jnp.max(peak, axis=2, keepdims=True)
    selection = jnp.array([0.]*5+[1.]*23)[None,None,:]
    weights = temporal[None,:,None]*audible*selection
    # Power groups penalize missing upper-band energy as well as the shape of
    # individual quiet ridges. They have no spectral-convergence loud-bin bias.
    group_masks = []
    for note in notes:
        groups = np.zeros((4, len(frequencies)))
        groups[0] = abs(frequencies-note['frequency']) < max(35., .22*note['frequency'])
        groups[1] = (frequencies >= 5.5*note['frequency'])&(frequencies < 12.5*note['frequency'])
        groups[2] = (frequencies >= 12.5*note['frequency'])&(frequencies < 25.5*note['frequency'])
        groups[3] = (frequencies >= 4000)&(frequencies < 12000)
        group_masks.append(groups)
    group_masks = jnp.array(group_masks)
    group_reference = jnp.sqrt(jnp.einsum('ntf,ngf->ntg', spectrum**2, group_masks)+1e-20)
    return masks, envelopes, peak*.03+1e-6, weights, group_masks, group_reference, temporal


def metallic_term(predicted, prepared):
    """Log-envelope mismatch for partials 6–25 and sustained 4–12 kHz energy."""
    masks, reference, floor, weights, groups, group_reference, temporal = prepared
    spectrum = magnitude(predicted, 2048)
    estimate = jnp.einsum('ntf,npf->ntp', spectrum, masks)
    # Remove note level using the fundamental; the renderer also fixes RMS.
    ref_ratio = (reference+floor)/(reference[:,:,:1]+floor[:,:,:1])
    est_ratio = (estimate+floor)/(estimate[:,:,:1]+floor[:,:,:1])
    error = jnp.abs(jnp.log(est_ratio)-jnp.log(ref_ratio))
    individual = jnp.sum(error*weights)/(jnp.sum(weights)+1e-20)
    group_estimate = jnp.sqrt(jnp.einsum('ntf,ngf->ntg', spectrum**2, groups)+1e-20)
    group_floor = .01*jnp.max(group_reference, axis=1, keepdims=True)+1e-6
    ref = (group_reference+group_floor)/(group_reference[:,:,:1]+group_floor[:,:,:1])
    est = (group_estimate+group_floor)/(group_estimate[:,:,:1]+group_floor[:,:,:1])
    grouped = jnp.sum(jnp.abs(jnp.log(est[:,:,1:])-jnp.log(ref[:,:,1:]))*temporal[None,:,None])/(len(reference)*3*jnp.sum(temporal)+1e-20)
    return .25*individual+grouped


def ridge_prepare(target, notes):
    """Resolve nonharmonic mid-band ridges instead of rewarding blurred energy.

    The seven short STFTs can trade a strong stationary body line against nearby
    moving string partials. A 171-ms window separates these lines; a one-sided
    excess term leaves missing source extras to explicit instrument hypotheses.
    """
    reference = magnitude(target, 8192)
    f = np.fft.rfftfreq(8192, 1/48000)
    t = np.arange(reference.shape[1])*2048/48000
    masks = []
    for note in notes:
        distance = np.min(abs(f[:,None]-note['frequency']*np.arange(1,100)[None,:]),axis=1)
        masks.append((f>=700)&(f<=5000)&(distance>35))
    mask = jnp.array(masks)[:,None,:]*jnp.array((t>=.04)&(t<=.4))[None,:,None]
    floor = .001*jnp.max(reference,axis=(1,2),keepdims=True)+1e-5
    return reference, mask, floor


def ridge_term(predicted, prepared):
    reference, mask, floor = prepared
    excess = jnp.maximum(jnp.log(magnitude(predicted,8192)+floor)-jnp.log(reference+floor),0.)
    return jnp.sum(excess**2*mask)/(jnp.sum(mask)+1e-8)


def objective(predicted, prepared, p, base, sparsity=.003):
    spectral, envelope = terms(predicted, prepared)
    # Gain penalty is dimensionless relative to each initialized mode.
    penalty = sparsity*jnp.mean(jnp.exp(jnp.clip(p['mode_gain'], -15, 6))/jnp.array(base[2]))
    # Weak bounded prior on frequency shifts limits mode-swapping ambiguity.
    prior = .002*jnp.mean(p['mode_frequency']**2)+.0002*jnp.mean(p['detune']**2)
    return spectral+.25*envelope+penalty+prior
