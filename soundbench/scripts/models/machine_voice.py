"""Deterministic pitch correction, cross-synthesis and score-independent sob lanes.

The only source interface is models.voice.render: no audio pitch shifter and
no dependence on source or tract internals. All returned controls use caller
seconds, with engine pre-roll removed.
"""
from __future__ import annotations
import math
import copy
from pathlib import Path
import tomllib

import numpy as np
from numba import njit
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, lfilter, resample_poly, sosfilt, fftconvolve
from . import voice

CONFIG = Path(__file__).resolve().parents[2] / 'instruments/models/machine-voice.toml'
SR = 48000
RANGES = {
    'strength': (0,1), 'retune_ms': (0,1000), 'hysteresis_cents': (0,100),
    'glitch': (0,1), 'bands': (4,64), 'low_hz': (20,1000), 'high_hz': (1000,20000),
    'attack_ms': (.1,1000), 'release_ms': (.1,2000), 'formant_shift': (-24,24),
    'sibilance': (0,1), 'buzz_shape': (0,1), 'buzz_detune_cents': (0,100),
    'lowpass_hz': (100,20000), 'highpass_hz': (0,2000), 'drive': (0,1),
    'breakup': (0,1), 'mix': (0,1), 'seed': (0,2**32-1), 'sob': (0,1),
    'carrier_noise': (0,1),
}
CHOICES = {'mode': ('none','channel','lpc'), 'carrier': ('voice','buzz','piano','organ','audio')}
KEYS = set(RANGES) | set(CHOICES) | {'scale'}


def validate(config):
    if set(config) != KEYS:
        raise ValueError(f'machine_voice parameters must be {sorted(KEYS)}; unknown/missing: {sorted(set(config)^KEYS)}')
    for key, (low, high) in RANGES.items():
        v = config[key]
        if type(v) not in (int,float) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f'machine_voice.{key} must be finite within {low}..{high}')
    for key in ('bands','seed'):
        if type(config[key]) is not int:
            raise ValueError(f'machine_voice.{key} must be an integer')
    for key, choices in CHOICES.items():
        if config[key] not in choices:
            raise ValueError(f'machine_voice.{key} must be one of {choices}')
    scale = config['scale']
    if not isinstance(scale, (list,tuple)) or not scale or any(type(v) is not int or not 0 <= v <= 11 for v in scale) or len(set(scale)) != len(scale):
        raise ValueError('scale must contain distinct integer MIDI pitch classes 0..11')
    if config['low_hz'] >= config['high_hz'] or config['highpass_hz'] >= config['lowpass_hz']:
        raise ValueError('machine_voice filter edges must be ordered')
    return config


def load_config(path=CONFIG, preset='corrected', overrides=None):
    data = tomllib.loads(Path(path).read_text())
    if set(data) != {'defaults','presets'} or set(data['presets']) != {'corrected','vocoder','talkbox','speaking-piano'}:
        raise ValueError('machine-voice tables must be defaults and the four named presets')
    validate(data['defaults'])
    for name, values in data['presets'].items():
        validate({**data['defaults'], **values})
    if preset not in data['presets']:
        raise ValueError(f'unknown machine_voice preset {preset!r}')
    return validate({**data['defaults'], **data['presets'][preset], **(overrides or {})})


def grid(scale):
    return np.array([12*octave + pc for octave in range(-4,16) for pc in sorted(scale)], float)


def snap(pitch, scale):
    g = grid(scale)
    p = np.asarray(pitch)
    return g[np.abs(p[...,None]-g).argmin(axis=-1)]


def correct_f0(t, f0, config):
    """Hysteretic target selection and one-pole retuning, preserving residual vibrato.

    Retuning smooths the quantized target, not the residual pitch error. This
    leaves (1-strength) of a held note's vibrato, even with a slow retune.
    """
    validate(config)
    t, f0 = np.asarray(t,float), np.asarray(f0,float)
    if t.ndim != 1 or f0.shape != t.shape or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(f0).all() or np.any(f0<=0) or np.any(np.diff(t)<=0):
        raise ValueError('correction needs matching finite positive f0 and increasing time arrays')
    p = 69 + 12*np.log2(f0/440)
    g = grid(config['scale'])
    nearest = np.abs(p[:,None]-g).argmin(axis=1)
    targets = np.empty(len(p)); glitches = np.zeros(len(p))
    index = nearest[0]
    rng = np.random.default_rng(config['seed'])
    for i in range(len(p)):
        candidate = nearest[i]
        if candidate != index:
            direction = np.sign(g[candidate]-g[index])
            midpoint = (g[index]+g[index+int(direction)])/2
            if direction*(p[i]-midpoint) > config['hysteresis_cents']/100:
                amount = max(config['glitch'], config['breakup'])
                duration = rng.uniform(.020,.040)
                end = np.searchsorted(t,t[i]+duration)
                glitches[i:end] += -direction*rng.uniform(40,60)*amount * np.sin(np.pi*(t[i:end]-t[i])/duration)**2 / 100
                index = candidate
        targets[i] = g[index]
    if config['retune_ms']:
        # Leaps retune faster: the time constant shrinks as 2/|leap| beyond a
        # whole tone, so a sobbed fall of a fourth lands in tens of ms.
        dt = np.median(np.diff(t))
        smooth = np.empty(len(targets)); y = targets[0]; start = y
        for i, target in enumerate(targets):
            if i and target != targets[i-1]:
                start = y
            tau = config['retune_ms']/1000*min(1, 2/max(abs(target-start), 1e-9))
            y += (1-np.exp(-dt/tau))*(target-y)
            smooth[i] = y
    else:
        smooth = targets
    corrected = config['strength']*smooth + (1-config['strength'])*p + glitches
    return 440*2**((corrected-69)/12)


GESTURES = {
    'cry': {'from_cents': (-100,-1200,0), 'rise_ms': (150,1,2000), 'swell_db': (9,0,24)},
    'break': {'interval': (-5,-24,0), 'fall_ms': (40,1,1000), 'drop_db': (-16,-40,0)},
    'catch': {'depth_cents': (-250,-1200,0), 'ms': (150,1,2000)},
    'breath': {'ms': (90,1,1000)},
    'waver': {'amount': (1,0,1)},
    'heave': {'amount': (1,0,1), 'rate_hz': (2.2,.5,6), 'dips_per_s': (.8,0,3), 'dip_db': (-12,-24,0)},
    'sob': {'amount': (1,0,1), 'delay_ms': (0,0,20000), 'rise_cents': (0,-1200,0), 'rise_ms': (150,1,2000),
            'swell_db': (0,0,12), 'fall_cents': (0,-1200,0), 'fall_ms': (250,1,2000), 'tail': (0,0,1)},
}
# The held-note behaviour `sob` morphs between, measured on a reference
# recording of corrected singing: a normal hold wobbles at 5-6 Hz
# over about 30 cents peak to peak; a sobbed hold at 2.4 Hz over 224 cents, noisier.
SOB_RATE_HZ = (5.5, 2.4)
SOB_DEPTH_CENTS = (30., 205.)


def validate_gesture(gesture):
    kind = gesture.get('kind')
    if kind not in GESTURES or set(gesture)-({'kind','time','scale'} | set(GESTURES[kind])):
        raise ValueError(f'invalid voice gesture {gesture}')
    for key, (default,low,high) in GESTURES[kind].items():
        v = gesture.get(key,default)
        if type(v) not in (int,float) or not math.isfinite(v) or not low<=v<=high:
            raise ValueError(f'invalid {kind}.{key}: {low}..{high}')
    for key in ('time','scale'):
        v = gesture.get(key,1 if key=='scale' else None)
        if type(v) not in (int,float) or not math.isfinite(v) or v<0 or (key=='scale' and v>1):
            raise ValueError(f'invalid gesture {key}')


def sob_gestures(notes, amount, explicit=(), gap=.12, first=True):
    """Choose the highest note (longest breaks ties) in each contiguous phrase.

    Explicit requests replace all automatic requests on their addressed onset.
    A break is addressed to its high note and executes at that note's end.
    Every phrase but the track's first is preceded by a breath; `first=False`
    says these notes continue a track, so the first phrase breathes too.
    """
    if not 0<=amount<=1 or not math.isfinite(amount):
        raise ValueError('sob must be 0..1')
    for art in explicit:
        validate_gesture(art)
        if not any(abs(r['time']-art['time'])<1e-7 for r in notes):
            raise ValueError('voice gesture needs an exact note onset')
    rows = sorted(notes,key=lambda r:r['time'])
    phrases=[]
    for row in rows:
        if not phrases or row['time']-max(r['off'] for r in phrases[-1])>gap:
            phrases.append([])
        phrases[-1].append(row)
    auto=[]
    if amount:
        for i,phrase in enumerate(phrases):
            top=max(phrase,key=lambda r:(r['pitch'], r['off']-r['time']))
            auto.extend(dict(kind=kind,time=top['time'],scale=amount) for kind in ('cry','break'))
            for row in phrase:
                if row['off']-row['time']>1.5:
                    auto.append(dict(kind='catch',time=row['time'],scale=amount))
                if row['off']-row['time']>1.0:
                    auto.append(dict(kind='heave',time=row['time'],scale=amount))
            if i or not first:
                auto.append(dict(kind='breath',time=phrase[0]['time'],scale=amount))
    return [a for a in auto if not any(abs(a['time']-e['time'])<1e-7 for e in explicit)] + list(explicit)


def sob_headroom(notes, seconds, amount=0, explicit=(), seed=1729, gap=.12):
    """The CC11 headroom a whole-track `sob_lanes` would reserve (no pitch reference render)."""
    return sob_lanes(notes,seconds,amount,explicit,(),seed,gap,headroom_only=True)[0]


def sob_lanes(notes, seconds, amount=0, explicit=(), midi_controls=(), seed=1729, gap=.12,
              voice_config=None, first=True, headroom=None, stream=0, headroom_only=False):
    """Expand gestures at 1 ms; expression composes multiplicatively.

    CC11 cannot exceed 127: reserve the largest requested cry's headroom on
    the whole track, rather than clipping a swell. No gesture means no change.
    A passage rendered apart passes the whole track's `headroom` (see
    `sob_headroom`) so its levels match a whole-track render, and `stream`
    to draw its own random sequence; `first=False` marks it as a continuation.
    """
    gestures=sob_gestures(notes,amount,explicit,gap,first)
    if not gestures and not headroom:
        return (0. if headroom_only else list(midi_controls)), gestures
    t=np.arange(math.ceil(seconds*1000)+1)/1000
    rows=sorted(notes,key=lambda r:r['time'])
    idx=np.maximum(0,np.searchsorted([r['time'] for r in rows],t,side='right')-1)
    base=np.array([r['pitch'] for r in rows])[idx]
    if not headroom_only and any(a['kind'] in ('cry','break','sob') for a in gestures):
        # Absolute reaches/falls replace the voice's ordinary glide on their
        # active span. Ask the public voice interface for its unmodulated f0;
        # don't inspect source/tract controls or duplicate its pitch planner.
        reference_config=copy.deepcopy(voice_config or voice.load_config())
        reference_config['expression']['vibrato_cents']=0
        _, reference=voice.render(notes,seconds,reference_config,True)
        reference_t=reference['t']-reference_config['engine']['lead_seconds']
        base=69+12*np.log2(np.interp(t,reference_t,reference['f0'])/440)
    pitch=np.zeros(len(t)); db=np.zeros(len(t)); breath=np.zeros(len(t))
    given=headroom
    headroom=max((a.get('swell_db',9)*a.get('scale',1) for a in gestures if a['kind']=='cry'),default=0)
    rng=np.random.default_rng([seed,stream] if stream else seed)
    for art in sorted(gestures,key=lambda a:(a['time'],a['kind']=='break')):
        validate_gesture(art)
        row=next(r for r in rows if abs(r['time']-art['time'])<1e-7)
        on,off=row['time'],row['off']; scale=art.get('scale',1); kind=art['kind']
        params={k:art.get(k,v[0]) for k,v in GESTURES[kind].items()}
        if kind=='cry':
            m=(t>=on)&(t<off)
            u=np.clip((t[m]-on)/(params['rise_ms']/1000),0,1)
            ease=1-(1-u)**2
            pitch[m]+=(row['pitch']-base[m])*100 + params['from_cents']*(1-ease)*scale
            db[m]+=params['swell_db']*ease*scale
        elif kind=='break':
            next_on=next((r['time'] for r in rows if r['time']>off+1e-7),seconds)
            m=(t>=off)&(t<next_on)
            u=np.clip((t[m]-off)/(params['fall_ms']/1000),0,1)
            target=row['pitch']+params['interval']*u*scale
            pitch[m]+=(target-base[m])*100
            cry=next((a for a in gestures if a['kind']=='cry' and abs(a['time']-on)<1e-7),None)
            swell=cry.get('swell_db',9)*cry.get('scale',1) if cry else 0
            db[m]+=swell+params['drop_db']*u*scale
        elif kind=='catch':
            start=(on+off)/2; duration=params['ms']/1000
            m=(t>=start)&(t<start+duration)
            shape=np.sin(np.pi*(t[m]-start)/duration)**2
            pitch[m]+=params['depth_cents']*shape*scale
            db[m]-=6*shape*scale
        elif kind=='sob':
            # The note held as a sob: absolute pitch (replacing the voice's glide)
            # with a slow, wide, irregular wobble, breath rising into the tone, a
            # glottal catch at 40% of the hold when the sob is deep, and an
            # optional reach from below and fall landing on the note's end.
            amount=params['amount']*scale
            start=on+params['delay_ms']/1000
            m=np.flatnonzero((t>=start)&(t<off))
            if len(m)>2:
                u=t[m]-start; span=off-start
                rate=SOB_RATE_HZ[0]+(SOB_RATE_HZ[1]-SOB_RATE_HZ[0])*amount
                depth=SOB_DEPTH_CENTS[0]+(SOB_DEPTH_CENTS[1]-SOB_DEPTH_CENTS[0])*amount
                k=max(3,round(1000/(rate*4)))
                drift=np.convolve(rng.standard_normal(len(m)+2*k),np.hanning(k),'same')[k:-k]
                drift/=np.std(drift)+1e-12
                phase=2*np.pi*np.cumsum(rate*(1+.18*drift))/1000+2*np.pi*rng.random()
                swing=1+.25*np.convolve(rng.standard_normal(len(m)+2*k),np.hanning(k),'same')[k:-k]/(k**.5)
                taper=voice.smoothstep(u/.12)*voice.smoothstep((off-t[m])/.06)
                wobble=depth/2*np.sin(phase)*np.clip(swing,.6,1.4)*taper
                rise=np.clip(u/(params['rise_ms']/1000),0,1)
                ease=1-(1-rise)**2
                fall_u=np.clip((t[m]-(off-params['fall_ms']/1000))/(params['fall_ms']/1000),0,1)
                fall=params['fall_cents']*(fall_u*fall_u*(3-2*fall_u))
                pitch[m]+=(row['pitch']-base[m])*100+wobble+params['rise_cents']*(1-ease)+fall
                db[m]+=params['swell_db']*ease*scale
                headroom=max(headroom,params['swell_db']*scale)
                breath[m]=np.maximum(breath[m],45*amount*taper)
                if params['tail']:
                    tail=np.clip((u-.7*span)/(.3*span),0,1)
                    breath[m]=np.maximum(breath[m],127*params['tail']*tail**2)
                if amount>=.6 and span>.6:
                    centre=start+.4*span; dip=rng.uniform(.04,.07)
                    d=np.clip((t[m]-centre)/dip,0,1); dip_shape=np.sin(np.pi*d)**2
                    db[m]-=10*dip_shape
                    after=t[m]-(centre+dip)
                    sharp=rng.uniform(50,150)*np.exp(-np.clip(after,0,None)/.06)*(after>=0)
                    pitch[m]+=sharp*voice.smoothstep(np.clip(after,0,None)/.01)
        elif kind=='breath':
            start=max(0,on-params['ms']/1000)
            m=(t>=start)&(t<on)
            breath[m]=np.maximum(breath[m],127*scale)
        elif kind=='heave':
            # Sobbing in the level: a slow irregular heave (3.2 dB SD at full
            # amount) and sparse deep dips where breath replaces voice, as
            # measured on the reference's holds.
            m=np.flatnonzero((t>=on)&(t<off))
            if len(m)>2:
                k=round(1000/params['rate_hz'])
                noise=np.convolve(rng.standard_normal(len(m)+2*k),np.hanning(k),'same')[k:-k]
                noise/=np.std(noise)+1e-12
                taper=voice.smoothstep((t[m]-on)/.1)*voice.smoothstep((off-t[m])/.1)
                level=3.2*params['amount']*scale*noise*taper
                for _ in range(rng.poisson(params['dips_per_s']*(off-on))):
                    width=rng.uniform(.08,.15); centre=rng.uniform(on+width,max(on+width,off-width))
                    shape=np.sin(np.pi*np.clip((t[m]-centre+width/2)/width,0,1))**2
                    level+=params['dip_db']*params['amount']*scale*shape
                    pitch[m]-=30*params['amount']*scale*shape
                    breath[m]=np.maximum(breath[m],60*params['amount']*scale*shape)
                db[m]+=level
                headroom=max(headroom,float(level.max()))
        else:
            m=(t>=on)&(t<off)
            # Smooth irregular 5–7 Hz motion, deterministic per gesture.
            phase=2*np.pi*rng.random()
            rate=6+np.sin(2*np.pi*.71*t[m]+phase)
            wave=np.sin(2*np.pi*np.cumsum(rate)/1000+phase)
            taper=voice.smoothstep((t[m]-on)/.05)*voice.smoothstep((off-t[m])/.05)
            pitch[m]+=35*params['amount']*scale*wave*taper
            db[m]+=2*params['amount']*scale*wave*taper
            headroom=max(headroom,2*params['amount']*scale)
    if headroom_only:
        return headroom, gestures
    if given is not None:
        headroom=given
    output=[]
    for control,default in ((11,127),('pitch_cents',0),(2,0)):
        lane=sorted((r for r in midi_controls if r['control']==control),key=lambda r:r['time'])
        if any(not math.isfinite(r['time']) or r['time']<0 or not math.isfinite(r['value']) or
               (control!='pitch_cents' and not 0<=r['value']<=127) for r in lane):
            raise ValueError('invalid input sob controller')
        original=np.interp(t,[r['time'] for r in lane],[r['value'] for r in lane]) if lane else np.full(len(t),default)
        values=original*10**((db-headroom)/20) if control==11 else original+pitch if control=='pitch_cents' else np.maximum(original,breath)
        if control==11:
            values=np.minimum(values,127)
        output.extend(dict(time=float(at),control=control,value=float(v)) for at,v in zip(t,values))
    output.extend(r for r in midi_controls if r['control'] not in (11,2,'pitch_cents'))
    return output, gestures


@njit(cache=True)
def _envelope(x, attack, release):
    """Asymmetric one-pole follower; recurrence requires a sample loop."""
    out=np.empty(len(x)); state=0.
    for i in range(len(x)):
        value=abs(x[i])
        state+=(attack if value>state else release)*(value-state)
        out[i]=state
    return out


def rms(x):
    return np.sqrt(np.mean(np.asarray(x)**2)) if len(x) else 0.


def makeup(x, reference):
    return x * rms(reference)/max(rms(x),1e-12)


def buzz(f0, config):
    """Harmonic additive saw/pulse blend with Nyquist taper (no aliased edges)."""
    out=np.zeros(len(f0)); phase=2*np.pi*np.cumsum(f0)/SR
    for detune in (0,config['buzz_detune_cents']):
        phi=phase*2**(detune/1200)
        freq=f0*2**(detune/1200)
        for h in range(1,min(512,int(SR/(2*freq.min())))+1):
            weight=np.clip((SR/2-h*freq)/(SR*.03),0,1)
            amp=((1-config['buzz_shape'])+config['buzz_shape']*2*np.sin(np.pi*h*.15))/h
            out+=amp*np.sin(h*phi)*weight/2
    return out


def _lpc(frame, order=24):
    frame=frame-frame.mean()
    r=fftconvolve(frame,frame[::-1],mode='full')[len(frame)-1:len(frame)+order]
    if r[0]<1e-18:
        return np.r_[1.,np.zeros(order)]
    r[0]*=1.001  # diagonal loading bounds resonances on sparse harmonic frames
    a=np.r_[1.,-solve_toeplitz(r[:order],r[1:order+1])]
    # Bandwidth expansion makes the all-pole envelope safely damped.
    return a*.995**np.arange(order+1)


def whiten(x):
    """LPC inverse filtering, Hann overlap-add; retain the carrier's amplitude."""
    size,hop=1200,600
    padded=np.pad(x,(size,size)); out=np.zeros_like(padded); norm=np.zeros_like(padded)
    w=np.hanning(size)
    for start in range(0,len(padded)-size+1,hop):
        frame=padded[start:start+size]
        residual=lfilter(_lpc(frame*w),[1.],frame)
        out[start:start+size]+=residual*w
        norm[start:start+size]+=w
    result=(out/np.maximum(norm,1e-9))[size:size+len(x)]
    return makeup(result,x)


def channel(modulator, carrier, config, rng):
    """Band vocoder; bands stream in order, holding two envelopes rather than all of them."""
    edges=np.geomspace(config['low_hz'],config['high_hz'],config['bands']+1)
    centers=np.sqrt(edges[:-1]*edges[1:])
    attack=1-np.exp(-1/(SR*config['attack_ms']/1000))
    release=1-np.exp(-1/(SR*config['release_ms']/1000))
    filters=[butter(4,[lo,hi],'bandpass',fs=SR,output='sos') for lo,hi in zip(edges[:-1],edges[1:])]
    n=len(centers); lows=max(1,n//2); highs=max(1,n//4)
    out=np.zeros(len(modulator))
    shift=config['formant_shift']/12*np.log(2)
    positions=np.interp(np.log(centers)-shift,np.log(centers),np.arange(n),left=-1,right=n)
    # Output band i reads envelopes int(pos) and the one above; both exist once
    # band `ready` is computed, since positions rise with i.
    pending=[i for i,pos in enumerate(positions) if 0<=pos<=n-1]
    envelopes={}; low=np.zeros(len(modulator)) if config['sibilance'] else None; high=np.zeros_like(low) if low is not None else None
    for j,filt in enumerate(filters):
        env=_envelope(sosfilt(filt,modulator),attack,release)
        if config['breakup']:
            env*=fault_lane(len(env),rng,config['breakup'])
        envelopes[j]=env
        if low is not None:
            if j<lows: low+=env
            if j>=n-highs: high+=env
        while pending and min(int(positions[pending[0]])+1,n-1)<=j:
            i=pending.pop(0); pos=positions[i]
            a=int(pos); b=min(a+1,n-1); u=pos-a
            band=sosfilt(filters[i],carrier)
            # Carrier normalization is global, so piano decay remains audible.
            out+=band*((1-u)*envelopes[a]+u*envelopes[b])/max(rms(band),1e-9)
        keep=min((int(positions[i]) for i in pending),default=n)
        for old in [k for k in envelopes if k<keep]:
            del envelopes[old]
    if config['sibilance']:
        unvoiced=np.clip((high/(low+1e-9)-.4)/1.6,0,1)
        noise=sosfilt(butter(4,[max(1000,config['high_hz']/2),config['high_hz']], 'bandpass',fs=SR,output='sos'),rng.standard_normal(len(out)))
        out+=config['sibilance']*noise*high*unvoiced
    return out


def lpc(modulator, carrier, shift):
    """Order-24, 25 ms LPC envelope filtering in frequency, Hann overlap-add.

    Frequency-domain warping evaluates the all-pole envelope at shifted
    frequencies without moving the carrier's pitch or risking unstable poles.
    """
    size,hop,nfft=1200,600,2048
    mod=np.pad(modulator,(size,size)); car=np.pad(whiten(carrier),(size,size))
    out=np.zeros_like(mod); norm=np.zeros_like(mod); w=np.hanning(size)
    freqs=np.fft.rfftfreq(nfft,1/SR); ratio=2**(shift/12)
    for start in range(0,len(mod)-size+1,hop):
        frame=mod[start:start+size]*w
        envelope=1/np.maximum(np.abs(np.fft.rfft(_lpc(frame),nfft)),1e-6)
        envelope=np.interp(freqs/ratio,freqs,envelope,left=envelope[0],right=0)
        envelope/=max(np.sqrt(np.mean(envelope**2)),1e-9)
        filtered=np.fft.irfft(np.fft.rfft(car[start:start+size]*w,nfft)*envelope,nfft)[:size]
        # Modulator's loudness carries phrasing; piano's decay stays in car.
        out[start:start+size]+=filtered*w*rms(frame)
        norm[start:start+size]+=w*w
    return (out/np.maximum(norm,1e-9))[size:size+len(modulator)]


def fault_lane(samples, rng, amount):
    lane=np.ones(samples); at=0
    while at<samples:
        at+=int(rng.uniform(.15,.7)*SR)
        end=min(samples,at+int(rng.uniform(.02,.08)*SR))
        lane[at:end]=1-amount*rng.uniform(.4,1)
    return lane


def colour(audio, config, rng):
    out=audio.copy()
    if config['breakup']:
        out*=fault_lane(len(out),rng,config['breakup'])
        at=0
        while at<len(out):
            at+=int(rng.uniform(.25,.8)*SR)
            end=min(len(out),at+int(rng.uniform(.02,.08)*SR))
            if at>=end: break
            factor=int(2+30*config['breakup'])
            out[at:end]=(1-config['breakup'])*out[at:end]+config['breakup']*out[at:end][np.arange(end-at)//factor*factor]
    if config['drive'] and rms(out)>0:
        up=resample_poly(out,4,1)
        scale=max(rms(up)*3,1e-9); gain=1+12*config['drive']; bias=.15*config['drive']
        nonlinear=(np.tanh(up/scale*gain+bias)-np.tanh(bias))/gain
        saturated=(1-config['drive'])*up/scale+config['drive']*nonlinear
        out=makeup(resample_poly(saturated,1,4)[:len(out)],out)
    if config['highpass_hz']:
        out=sosfilt(butter(4,config['highpass_hz'],'highpass',fs=SR,output='sos'),out)
    # Limit after distortion so added harmonics don't escape the requested band.
    return sosfilt(butter(8,config['lowpass_hz'],'lowpass',fs=SR,output='sos'),out)


def process(corrected, f0, c, notes=(), carrier_audio=None, stream=0):
    """The machine after the source: cross-synthesis onto a carrier, then colour.

    corrected: the pitch-corrected voice (any source engine), f0 its per-sample
    pitch in Hz; notes are needed only by the piano carrier. stream separates
    the random draws of passages of one track rendered apart (0: the track's own).
    """
    key=[c['seed']]+([stream] if stream else [])
    samples=len(corrected)
    if c['carrier']=='voice':
        carrier=whiten(corrected) if c['mode']=='channel' else corrected
    elif c['carrier']=='buzz':
        carrier=buzz(f0,c)
        if c['carrier_noise']:
            # Noise between the buzz's partials, from 2 kHz up, so no pure
            # line whines in the ear's most sensitive band.
            rng=np.random.default_rng(np.random.SeedSequence(key+[12]))
            hiss=sosfilt(butter(2,2000,'highpass',fs=SR,output='sos'),rng.standard_normal(samples))
            hiss*=rms(sosfilt(butter(2,2000,'highpass',fs=SR,output='sos'),carrier))/max(rms(hiss),1e-12)
            carrier=carrier+c['carrier_noise']*hiss
    elif c['carrier']=='piano':
        from . import piano
        rows=[{**r,'pitch':float(snap(r['pitch'],c['scale']))} for r in notes]
        carrier=piano.render(rows,samples/SR).sum(axis=1)
    elif c['carrier']=='audio':
        carrier=np.asarray(carrier_audio,float)
        if carrier.shape!=(samples,) or not np.isfinite(carrier).all():
            raise ValueError('audio carrier needs a finite mono array matching the render duration')
    if c['mode']=='none':
        wet=corrected
    else:
        rng=np.random.default_rng(np.random.SeedSequence(key+[10]))
        wet=channel(corrected,carrier,c,rng) if c['mode']=='channel' else lpc(corrected,carrier,c['formant_shift'])
        wet=makeup(wet,corrected)
    return colour(wet,c,np.random.default_rng(np.random.SeedSequence(key+[11])))


def render(notes, seconds=None, config=None, voice_config=None, return_controls=False,
           midi_controls=(), gestures=(), carrier_audio=None,
           first=True, headroom=None, stream=0):
    c=validate(config or load_config()); vc=voice_config or voice.load_config()
    if c['carrier']=='organ':
        raise ValueError('organ carrier is not yet available')
    if seconds is None:
        seconds=max((r['off'] for r in notes),default=0)+vc['engine']['tail_seconds']
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError('machine voice duration must be positive and finite')
    samples=round(seconds*SR)
    if not notes:
        result=np.zeros(samples)
        return (result,None) if return_controls else result
    lanes,placed=sob_lanes(notes,seconds,c['sob'],gestures,midi_controls,c['seed'],
                           vc['expression']['legato_gap_s'],vc,first,headroom,stream)
    # Dry is the performed voice before correction and colour, with sob intact.
    dry,k=voice.render(notes,seconds,vc,True,midi_controls=lanes)
    capture={}
    def mapper(t,f):
        corrected=correct_f0(t,f,c)
        capture.update(t=t.copy(),input_f0=f.copy(),f0=corrected.copy())
        return corrected
    corrected,k=voice.render(notes,seconds,vc,True,midi_controls=lanes,f0_map=mapper)
    t=np.arange(samples)/SR
    f0=np.interp(t,capture['t'],capture['f0'])
    out=c['mix']*process(corrected,f0,c,notes,carrier_audio,stream)+(1-c['mix'])*dry
    if return_controls:
        # Preserve control geometry but expose only the public caller timeline.
        keep=(capture['t']>=0)&(capture['t']<=seconds)
        result={key:value[keep] for key,value in capture.items()}
        result.update(amplitude=k['amplitude'][keep],gestures=placed,midi_controls=lanes)
        return out,result
    return out
