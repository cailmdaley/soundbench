"""Gear-tuned continuous tonewheels, bounced contacts, scanner line and Leslie.

Waveforms are generated, never sampled. The 91-wheel bank spans C1–F#8;
high harmonics fold back by octaves. Wheel phases are shared across keys.
"""
import math
import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt, lfilter
from .keys_common import SR, ROOT, load_config as _load, validate, curve, smooth, finish
from soundbench.scripts.paths import preset_path

CONFIG = preset_path('organ')
GEARS = np.array([(85,104),(71,82),(67,73),(35,36),(69,67),(12,11),
                  (37,32),(49,40),(48,37),(11,8),(67,46),(54,35)], float)
OFFSETS = (-12, 7, 0, 12, 19, 24, 28, 31, 36)


def load_config(path=CONFIG):
    cfg = _load('organ', path)
    registration(cfg['drawbars']['registration'])
    if cfg['percussion']['harmonic'] not in (2,3):
        raise ValueError('percussion harmonic is 2 or 3')
    for k in ('fast_decay_s','slow_decay_s'):
        if not .02 <= cfg['percussion'][k] <= 5:
            raise ValueError('invalid percussion decay')
    for k in ('horn_ramp_s','drum_ramp_s'):
        if not .05 <= cfg['leslie'][k] <= 20:
            raise ValueError('invalid rotary ramp')
    return cfg


def registration(value):
    text = str(value)
    if len(text) != 9 or any(c not in '012345678' for c in text):
        raise ValueError('drawbar registration needs nine digits 0–8')
    return np.array([int(c) for c in text])


def wheel_pitch(pitch):
    while pitch < 24: pitch += 12
    while pitch > 114: pitch -= 12
    return pitch


def wheel_frequency(pitch):
    pitch = wheel_pitch(pitch)
    # 1200 rpm motor, two teeth on the lowest octave, octave tooth doublings.
    return 40*GEARS[pitch%12,0]/GEARS[pitch%12,1]*2**((pitch-24)//12)


def percussion_triggers(notes):
    """A simultaneous chord triggers once; overlap prevents capacitor recharge."""
    groups = {}
    for n in notes:
        groups.setdefault(n['time'], []).append(n)
    active_until = -1
    result = []
    for t, group in sorted(groups.items()):
        if t >= active_until:
            result.append((t, group))
        active_until = max(active_until, max(n['off'] for n in group))
    return result


def scanner(y, modes, cfg):
    """Sixteen dispersive line-box taps scanned in a triangular cycle.

    First-order allpass sections approximate the LC delay-line dispersion;
    scanner contacts crossfade neighboring taps. C mixes wet and dry paths.
    """
    s = cfg['scanner']; n = len(y)
    phase = (np.arange(n)/SR*s['rate_hz']) % 1
    triangle = 1-np.abs(2*phase-1)
    result = y.copy()
    tap = y.copy()
    # Tap delays grow toward low frequencies, as in a dispersive line box.
    a = s['allpass_coefficient']
    for mode in sorted(set(modes)-{0}):
        depth = ((mode-1)%3+1)/3
        position = triangle*15*depth
        lo = position.astype(int); u = position-lo
        wet = np.zeros(n)
        tap = y.copy()
        for j in range(17):
            if j:
                tap = lfilter([a,1], [1,a], tap)
            wet += tap*(np.where(lo == j, 1-u, 0)+np.where(lo+1 == j, u, 0))
        if mode >= 4: wet = .5*(wet+y)
        mask = modes == mode
        result[mask] = wet[mask]
    return result


@njit(cache=True)
def rotor(target, slow, fast, ramp, sr):
    rates = np.empty(len(target)); phase = np.empty(len(target))
    speed = slow; angle = 0.0; a = math.exp(-1/(ramp*sr))
    for i in range(len(target)):
        speed = a*speed+(1-a)*(slow+(fast-slow)*target[i]/127)
        angle += 2*math.pi*speed/sr
        rates[i] = speed; phase[i] = angle
    return rates, phase


def leslie(y, controls, cfg):
    l = cfg['leslie']; n = len(y)
    target = curve(controls, 68, n, l['speed'])
    low = sosfilt(butter(2,l['crossover_hz'],fs=SR,output='sos'),y)
    high = y-low
    out = np.zeros((n,2)); samples = np.arange(n)
    for band, name, direction in ((high,'horn',1),(low,'drum',-1)):
        _, phase = rotor(target,l[name+'_slow_hz'],l[name+'_fast_hz'],l[name+'_ramp_s'],SR)
        phase *= direction
        radius = l[name+'_radius_m']
        for ch, mic in enumerate((-.65,.65)):
            angle = phase-mic
            delay = (l['distance_m']-radius*np.cos(angle))/343*SR
            moved = np.interp(samples-delay,samples,band,left=0,right=0)
            amplitude = 1+l[name+'_am_depth']*np.cos(angle)
            # Rearward radiation and a stationary cabinet path avoid hard nulls.
            out[:,ch] += .78*moved*amplitude+.12*band
    return out


def render(notes, seconds, config=None, controls=()):
    cfg = config or load_config(); validate(notes,seconds,controls,36,96)
    n = round(seconds*SR); out = np.zeros(n)
    seed = cfg['engine']['seed']; rng = np.random.default_rng(seed)
    phases = np.random.default_rng(seed).uniform(0,2*np.pi,91)
    bars = registration(cfg['drawbars']['registration'])
    bar_curves = [smooth(curve(controls,12+j,n,b*127/8),.004)*8/127 for j,b in enumerate(bars)]
    enabled = curve(controls,80,n,127 if cfg['percussion']['enabled'] else 0) >= 64
    click = cfg['contacts']['click_gain']; release = cfg['contacts']['release_s']
    for note in sorted(notes,key=lambda x:(x['time'],x['pitch'],x['off'],x['velocity'])):
        start = round(note['time']*SR); off = round(note['off']*SR)
        end = min(n,off+round(release*12*SR)); t = np.arange(start,end)/SR
        env = np.ones(len(t)); r = np.arange(end-start)
        env *= np.minimum(1,r/(SR*.002))
        env *= np.exp(-np.maximum(r-(off-start),0)/(SR*release))
        for j, offset in enumerate(OFFSETS):
            p = wheel_pitch(note['pitch']+offset); hz = wheel_frequency(p)
            if hz > SR*.45: continue
            contact = round(rng.uniform(0,.0018)*SR)
            gate = np.ones(len(t)); gate[:contact] = 0
            # Alternating closed/open microcontacts settle within 3 ms.
            edge = contact; state = 1
            while edge < min(len(t),contact+round(.003*SR)):
                next_edge = min(len(t),edge+max(1,round(rng.exponential(.00018)*SR)))
                gate[edge:next_edge] = state
                state = 1-state; edge = next_edge
            e = env*smooth(gate,.00012)
            amp = np.where(bar_curves[j][start:end]>0,2**((bar_curves[j][start:end]-8)*.5),0)
            # B-3 percussion steals the ninth contact (1-foot drawbar).
            if j == 8: amp = amp*(~enabled[start:end])
            tone = np.sin(2*np.pi*hz*t+phases[p-24])
            leak_p = wheel_pitch(p+1)
            tone += cfg['tonewheels']['leakage']*np.sin(2*np.pi*wheel_frequency(leak_p)*t+phases[leak_p-24])
            out[start:end] += .075*e*amp*tone
        # Stochastic electrical contact bounce, not velocity-scaled white attack.
        for when, amount in ((start,1),(off,.45)):
            count = min(round(.012*SR),n-when)
            if count <= 0: continue
            tt = np.arange(count)/SR
            noise = rng.normal(size=count)
            noise = sosfilt(butter(1,1400,'highpass',fs=SR,output='sos'),noise)
            out[when:when+count] += click*amount*noise*np.exp(-tt/cfg['contacts']['click_decay_s'])
    pc = cfg['percussion']
    triggers = [t for t,_ in percussion_triggers(notes)]
    for note in sorted(notes,key=lambda x:(x['time'],x['pitch'])):
        start_time = max(t for t in triggers if t <= note['time'])
        start = round(note['time']*SR)
        if not enabled[min(start,n-1)]: continue
        harmonic = 3 if curve(controls,81,n,127 if pc['harmonic']==3 else 0)[start]>=64 else 2
        fast = curve(controls,82,n,127 if pc['fast'] else 0)[start]>=64
        soft = curve(controls,83,n,127 if pc['soft'] else 0)[start]>=64
        tau = pc['fast_decay_s'] if fast else pc['slow_decay_s']
        end = min(n,round(note['off']*SR)+round(release*12*SR))
        t = np.arange(start,end)/SR; age = t-start_time
        envelope = (1-np.exp(-age/.001))*np.exp(-age/tau)
        envelope *= np.exp(-np.maximum(t-note['off'],0)/release)
        envelope *= np.minimum(1,np.maximum(t-note['time'],0)/.001)
        p = wheel_pitch(note['pitch']+(12 if harmonic==2 else 19))
        out[start:end] += pc['gain']*(.55 if soft else 1)*envelope*np.sin(2*np.pi*wheel_frequency(p)*t+phases[p-24])
    modes = np.rint(curve(controls,1,n,cfg['scanner']['mode']*127/6)*6/127).astype(int)
    out = scanner(out,modes,cfg)
    expression = smooth(curve(controls,11,n,127)/127,.01)
    drive = cfg['preamp']['drive']
    out = np.tanh(out*drive)/drive*expression
    return finish(leslie(out,controls,cfg) if cfg['leslie']['enabled'] else np.column_stack((out,out)),cfg['engine']['output_gain'])
