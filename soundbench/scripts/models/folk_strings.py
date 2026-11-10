"""Mandolin courses and lap resonator presets on the guitar waveguide engine.

Mandolin uses four primary/secondary pairs; dobro uses six polarized strings.
All synthesis is procedural. Event times and ramps use the guitar's 48 kHz grid.
"""
from __future__ import annotations
import math
import copy
import numpy as np
from scipy.signal import lfilter
from soundbench.scripts.models import guitar

ROOT = guitar.ROOT
SR = 48000


def load_config(instrument, path=None):
    if instrument not in ('mandolin', 'dobro'):
        raise ValueError('instrument is mandolin or dobro')
    c = guitar.load_config(path or ROOT/f'instruments/models/{instrument}.toml')
    if len(c['strings']['open_midi']) != (4 if instrument == 'mandolin' else 6):
        raise ValueError('wrong string count for preset')
    return c


def note(on, off, pitch, velocity, string, token, direction='down', **kwargs):
    """Explicitly stop even open courses: a score note-off means hand damping."""
    if direction not in ('down','up'):
        raise ValueError('pick direction is down or up')
    rows = guitar.note(on, off, pitch, velocity, string, token, **kwargs)
    if rows[0][1] == 0:
        rows[0][7] = int(direction == 'up')
    if not kwargs.get('let_ring', False):
        rows[1][8] = 2
    return rows


def tremolo(on, off, pitch, velocity, course, token, rate=12., evenness=.9,
            accent=1.12, seed=817, owner=0):
    """Alternate down/up picks; rate is attacks/s, evenness 0..1, accent 1..1.5."""
    if not all(math.isfinite(v) for v in (on, off, rate, evenness, accent)) or not 4 <= rate <= 24 or not 0 <= evenness <= 1 or not 1 <= accent <= 1.5 or off <= on:
        raise ValueError('invalid tremolo controls')
    rng = np.random.default_rng(seed+token)
    rows, t, i = [], on, 0
    phase = rng.uniform(0,2*np.pi)
    drift = 0.
    while t < off:
        drift = .92*drift+.08*rng.uniform(-1,1)
        drift_depth = (.02+.04*(1-evenness)) if evenness < 1 else 0.
        slow = drift_depth*(.7*np.sin(2*np.pi*.35*(t-on)+phase)+.3*drift)
        step = (1+slow+rng.uniform(-.18,.18)*(1-evenness))/rate
        gain = (accent if i % 2 == 0 else 1)*rng.uniform(1-.2*(1-evenness), 1)
        stroke = note(t, min(off,t+step), pitch, min(127,max(1,round(velocity*gain))), course, token+i, owner=owner,direction='up' if i%2 else 'down')
        # Only the last stroke releases; intervening picks sustain the course.
        stroke[0][8] = .15 if i % 2 == 0 else .17
        rows.append(stroke[0])
        if t+step >= off:
            rows.append(stroke[1])
        t += step
        i += 1
    return rows


def chop(on, pitches, courses, config, velocity=95, token=0, owner=0):
    """A 12 ms downstroke with a 20 ms left-hand stop on each course."""
    rows = guitar.strum(on,.075,pitches,config,velocity=velocity,strings=courses,
                        speed_ms=12,muted=True,token=token,owner=owner,let_ring=False)
    for row in rows:
        if row[1] == 0:
            row[7] = 0
        elif row[1] == 1:
            row[8] = 2
    return rows


def bar(on, points, string, token, owner=0):
    """Absolute MIDI-pitch control points: (seconds after on, pitch).

    The first point anchors an already plucked note; every next point ramps
    continuously from it. Fractional pitches and independent string slants work.
    """
    if len(points) < 2 or points[0][0] != 0:
        raise ValueError('bar points start at zero and need a destination')
    rows = []
    for (a, p), (b, q) in zip(points,points[1:]):
        if not all(math.isfinite(v) for v in (on,a,b,p,q)) or b <= a:
            raise ValueError('bar times must increase and pitches must be finite')
        rows.append([round((on+a)*SR),2,string-1,owner,q,0,max(1,round((b-a)*SR)),0,0,0,token])
    return rows


def vibrato_options(value=None):
    """Bar-rocking controls; depth in cents, delay/rise/minimum in seconds."""
    defaults = dict(depth_cents=18.,rate=5.5,onset_delay=.25,rise_seconds=.4,minimum_seconds=2/3)
    if value is False:
        defaults['depth_cents'] = 0.
    elif value is not None:
        if not isinstance(value,dict) or set(value)-set(defaults):
            raise ValueError('unknown vibrato controls')
        defaults.update(value)
    if not all(math.isfinite(v) for v in defaults.values()) or not 0 <= defaults['depth_cents'] <= 50 or not 1 <= defaults['rate'] <= 10 or not 0 <= defaults['onset_delay'] <= 5 or not .05 <= defaults['rise_seconds'] <= 2 or not 0 <= defaults['minimum_seconds'] <= 10:
        raise ValueError('invalid vibrato controls')
    return defaults


def express_bar(events, config, vibrato=None):
    """Compose growing vibrato with pitch ramps; open strings stay unmodulated.

    Long fretted/bar-sounded notes receive it by default. Existing wheel curves
    are the center pitch; disable automatic vibrato for hand-written oscillation.
    """
    controls = vibrato_options(vibrato)
    rows = [list(row) for row in events]
    if not controls['depth_cents']:
        return rows
    attacks = sorted((row for row in rows if row[1] in (0,3,4)),key=lambda r:r[0])
    for attack in attacks:
        token, lane, on = int(attack[10]), int(attack[2]), attack[0]
        ends = [r[0] for r in rows if r[10]==token and r[1] in (1,5) and r[0]>on]
        ends += [r[0] for r in attacks if r[2]==lane and r[0]>on]
        if not ends:
            continue
        off = min(ends)
        if (off-on)/SR <= controls['minimum_seconds']:
            continue
        ramps = sorted((r for r in rows if r[10]==token and r[1]==2 and on <= r[0]<off),key=lambda r:r[0])
        opening = config['strings']['open_midi'][lane]
        if max([attack[4]]+[r[4] for r in ramps]) <= opening+.01:
            continue
        settle, previous = on, attack[4]
        for ramp in ramps:
            if abs(ramp[4]-previous) >= .5:
                settle = max(settle,min(off,ramp[0]+ramp[6]))
            previous = ramp[4]
        start = settle+controls['onset_delay']*SR
        if start >= off:
            continue
        ticks = np.unique(np.r_[np.arange(on,off,SR/100),off,
            [r[0] for r in ramps],[min(off,r[0]+r[6]) for r in ramps]]).astype(int)
        center = np.full(len(ticks),attack[4],dtype=float)
        previous = attack[4]
        for ramp in ramps:
            use = ticks >= ramp[0]
            frac = np.clip((ticks[use]-ramp[0])/max(1,ramp[6]),0,1)
            if ramp[7]:frac = frac*frac*(3-2*frac)
            center[use] = previous+(ramp[4]-previous)*frac
            previous = ramp[4]
        age = (ticks-start)/SR
        growth = np.clip(age/controls['rise_seconds'],0,1)
        growth = growth*growth*(3-2*growth)
        fade = np.clip((off-ticks)/(.08*SR),0,1)
        offset = controls['depth_cents']/100*growth*fade*np.sin(2*np.pi*controls['rate']*np.maximum(age,0))
        pitches = np.clip(center+offset,opening,opening+config['strings']['max_fret'])
        rows = [r for r in rows if not (r[10]==token and r[1]==2)]
        for a,b,p in zip(ticks[:-1],ticks[1:],pitches[1:]):
            rows.append([int(a),2,lane,attack[3],float(p),0,int(b-a),0,0,0,token])
    return sorted(rows,key=lambda r:(r[0],0 if r[1]==1 else 1))


def vibrato(on, off, pitch, string, token, depth_cents=18., rate=5.5,
            onset_delay=.25, rise_seconds=.4, owner=0):
    """Explicit bar-vibrato articulation, independent of automatic-note threshold."""
    controls = vibrato_options(dict(depth_cents=depth_cents,rate=rate,onset_delay=onset_delay,rise_seconds=rise_seconds))
    times = np.unique(np.r_[np.arange(0,off-on,.01),off-on])
    if off <= on:
        raise ValueError('vibrato requires a positive duration')
    growth = np.clip((times-onset_delay)/rise_seconds,0,1)
    growth = growth*growth*(3-2*growth)
    fade = np.clip((off-on-times)/.08,0,1)
    points = list(zip(times,pitch+depth_cents/100*growth*fade*np.sin(2*np.pi*rate*np.maximum(times-onset_delay,0))))
    return bar(on,points,string,token,owner)


def render(events, seconds, owners=1, config=None, instrument='mandolin', vibrato=None):
    c = config or load_config(instrument)
    contact_events = events
    if instrument == 'dobro':
        events = express_bar(events,c,vibrato)
    local = copy.deepcopy(c)
    if instrument == 'dobro':
        local['excitation']['slide_noise'] = 0.
    audio = guitar.render(events, seconds, owners, local)
    if instrument != 'dobro' or not c['excitation']['slide_noise']:
        return audio
    # Colored friction energy scales with bar speed; it stops when the bar stops.
    current, active = {}, {}
    noise = np.zeros_like(audio)
    for row in sorted(contact_events,key=lambda r:r[0]):
        token = int(row[10])
        if row[1] in (0,3,4):
            current[token] = row[4]
            active[int(row[2])] = token
        elif row[1] == 2 and token in current and active.get(int(row[2])) == token:
            delta = abs(row[4]-current[token])
            current[token] = row[4]
            length = int(row[6])
            start, owner = int(row[0]), int(row[3])
            end = min(noise.shape[1],start+length)
            if end <= start or delta == 0:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([c['engine']['seed'],token,start]))
            burst = lfilter([1,-1],[1,-.85],rng.normal(size=end-start))
            burst *= np.sin(np.pi*(np.arange(end-start)+.5)/length)**2
            speed = min(2., math.sqrt(delta/max(length/SR,.001)/12))
            noise[owner,start:end] += burst*c['excitation']['slide_noise']*.02*speed*c['engine']['output_gain']
            # A light resonant squeak belongs to travel, not tiny vibrato segments.
            if delta >= .5 and length >= .04*SR:
                squeak = lfilter([1.,0.,-1.],[1.,-1.72,.9],rng.normal(size=end-start))
                squeak *= np.sin(np.pi*(np.arange(end-start)+.5)/length)**2
                squeak /= max(np.sqrt(np.mean(squeak*squeak)),1e-12)
                noise[owner,start:end] += squeak*c['excitation']['slide_noise']*.025*speed*c['engine']['output_gain']
    return guitar._output_guard(audio+noise,.85,.98)
