"""Four fretless stiff strings in a truncated modal basis, driving a large body.

Exact damped rotations integrate the string modes; unilateral fingerboard
contact projects displacement and reflects only inward contact velocity.
No recorded waveforms enter synthesis. String numbers 1..4 are G2 D2 A1 E1.
"""
from __future__ import annotations
import math
from pathlib import Path
import tomllib
import numpy as np
from numba import njit
from scipy.signal import lfilter
from soundbench.scripts.models.banjo import _output_guard

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'instruments/models/bass-gut.toml'
KINDS = {'pizz': 0, 'release': 1, 'slide': 2, 'slap': 3, 'dead': 4, 'ghost': 5, 'damp': 6, 'slap-back': 7}


def load_config(path=CONFIG):
    c = tomllib.loads(Path(path).read_text())
    if set(c) != {'engine', 'strings', 'excitation', 'body', 'fingerboard'}:
        raise ValueError('expected engine, strings, excitation, body, fingerboard')
    for table in c.values():
        for value in table.values():
            if not np.all(np.isfinite(np.asarray(value, dtype=float))):
                raise ValueError('parameters must be finite')
    e, s, x, b, f = (c[k] for k in ('engine','strings','excitation','body','fingerboard'))
    if e['sample_rate'] != 48000 or e['output_gain'] <= 0 or not 0.5 <= e['tail_seconds'] <= 5 or type(e['seed']) is not int or e['seed'] < 0:
        raise ValueError('invalid engine')
    if type(s['modes']) is not int or not 8 <= s['modes'] <= 96:
        raise ValueError('invalid mode count')
    for key in ('open_midi','length_m','tension_n','linear_density_kg_m','stiffness','gain'):
        if np.shape(s[key]) != (4,) or min(s[key]) <= 0:
            raise ValueError(f'strings.{key} needs four positive values')
    if not 0 < s['max_semitones'] <= 36 or not 0 < s['release_t60'] <= 1 or not 0 < s['dead_t60'] <= .3:
        raise ValueError('invalid neck/release')
    if not np.allclose(np.sqrt(np.array(s['tension_n'])/s['linear_density_kg_m'])/(2*np.array(s['length_m'])),440*2**((np.array(s['open_midi'])-69)/12),rtol=.002):
        raise ValueError('tension, density and length must agree with open tuning')
    if np.shape(s['decay_t60']) != (4,len(s['decay_hz'])) or min(np.ravel(s['decay_t60'])) <= 0 or min(s['decay_hz']) <= 0 or np.any(np.diff(s['decay_hz']) <= 0):
        raise ValueError('invalid string decay curves')
    if not 0 < x['position'] < .5 or not 0 < x['contact_ms'] <= 15 or not 0 < x['lowpass_hz'] < 20000 or not 0 <= x['intonation_cents'] <= 15 or not 0 < x['displacement_m'] <= .01 or not 0 < x['velocity_exponent'] <= 3:
        raise ValueError('invalid excitation')
    for key,low,high in [('attack_gain',0,8),('attack_t60',.06,.6),('attack_loss_slope',.2,2),('release_noise_gain',0,.8),('release_noise_ms',1,15),('attack_velocity_power',0,10)]:
        if not low <= x.get(key,low) <= high:
            raise ValueError(f'invalid excitation.{key}')
    if not 1.5 <= x.get('flesh_contact_ms',3.) <= 10:
        raise ValueError('flesh contact must last 1.5..10 ms')
    if not 0 <= x['noise_gain'] <= .1 or not 0 <= x['position_jitter'] <= .1:
        raise ValueError('invalid contact noise/jitter')
    if not len(b['weights']) or not (len(b['frequency_hz']) == len(b['t60_seconds']) == len(b['weights'])) or min(b['frequency_hz']) < 20 or max(b['frequency_hz']) >= 20000 or min(b['t60_seconds']) <= 0 or max(b['t60_seconds']) > .4 or min(b['weights']) < 0 or b['direct'] < 0:
        raise ValueError('invalid body modes')
    if not 0 < f['position'] < 1 or not 0 < f['clearance_m'] < .02 or not 0 <= f['restitution'] <= 1 or not 1 <= f['slap_displacement_multiplier'] <= 8 or not 0 <= f['impact_gain'] <= .1:
        raise ValueError('invalid contact')
    return c


def assign_string(pitch, config, explicit=None, sounding=None, started=None):
    if explicit is not None and (type(explicit) is not int or explicit not in range(1,5)):
        raise ValueError('string must be 1..4 (G D A E)')
    choices = [explicit-1] if explicit else range(4)
    choices = [s for s in choices if 0 <= pitch-config['strings']['open_midi'][s] <= config['strings']['max_semitones']]
    if not choices:
        raise ValueError(f'MIDI {pitch} is outside the selected bass string')
    free = [s for s in choices if sounding is None or not sounding[s]]
    return min(free, key=lambda s: pitch-config['strings']['open_midi'][s]) if free else min(choices,key=lambda s: started[s])


@njit(cache=True)
def _strings(rows, amplitudes, noises, samples, owners, sr, opens, stiffness, knots, times, gains, release, dead, fbpos, gap, restitution, impact_gain, contact_samples, attack_amplitudes, attack_t60, attack_slope):
    modes = amplitudes.shape[1]
    q, v = np.zeros((4,modes)), np.zeros((4,modes))
    fast_q,fast_v = np.zeros((4,modes)),np.zeros((4,modes))
    fast_excitation = np.zeros((4,modes))
    fast_decay = np.zeros((4,modes))
    omega = np.zeros((4,modes))
    co = np.zeros((4,modes))
    si = np.zeros((4,modes))
    decay = np.zeros((4,modes))
    pitch, source, target = opens.copy(), opens.copy(), opens.copy()
    start, length, shape = np.zeros(4), np.ones(4), np.zeros(4)
    tokens = np.full(4,-1,np.int64)
    owner = np.zeros(4,np.int64)
    loss = np.ones(4)
    colliding = np.zeros(4,np.bool_)
    active = np.zeros(4,np.bool_)
    dirty = np.zeros(4,np.bool_)
    excitation = np.zeros((4,modes))
    excitation_start = np.zeros(4)
    contact = np.sin(np.pi*fbpos*np.arange(1,modes+1))
    norm = np.sum(contact**2)
    output = np.zeros((owners,samples))
    impulse = np.zeros((owners,samples))
    event = 0
    for n in range(samples):
        while event < len(rows) and rows[event,0] <= n:
            r = rows[event]
            k, s, o, token = int(r[1]), int(r[2]), int(r[3]), int(r[10])
            if k in (0,3,4,5,7):
                q[s],v[s] = 0.,0.
                excitation[s] = amplitudes[event]
                fast_q[s],fast_v[s] = 0.,0.
                fast_excitation[s] = attack_amplitudes[event]
                excitation_start[s] = n
                pitch[s] = source[s] = target[s] = r[4]
                start[s], length[s] = n, 1
                tokens[s], owner[s], active[s] = token, o, True
                loss[s] = math.exp(-math.log(1000)/(dead*sr)) if k in (4,5,7) else 1.
                colliding[s] = k in (3,7)
                for j in range(noises.shape[1]):
                    if n+j < samples:
                        impulse[o,n+j] += noises[event,j]
                dirty[s] = True
            elif tokens[s] == token and owner[s] == o:
                if k in (1,6):
                    loss[s] = math.exp(-math.log(1000)/((release if k == 1 else .035)*sr))
                elif k == 2:
                    source[s], target[s] = pitch[s], r[4]
                    start[s], length[s], shape[s] = n, max(1,r[6]), r[7]
                    # Sliding never undamps an already released note.
            event += 1
        for s in range(4):
            if not active[s]:
                continue
            frac = min(1.,(n-start[s])/length[s])
            if shape[s] == 1:
                frac = frac*frac*(3-2*frac)
            newpitch = source[s]+frac*(target[s]-source[s])
            if dirty[s] or abs(newpitch-pitch[s]) > 1e-10:
                dirty[s] = False
                pitch[s] = newpitch
                f0 = 440*2**((pitch[s]-69)/12)
                # Stopping shortens the string, increasing relative stiffness.
                B = stiffness[s]*2**((pitch[s]-opens[s])/6)
                for j in range(modes):
                    h = j+1
                    f = f0*h*math.sqrt((1+B*h*h)/(1+B))
                    omega[s,j] = 2*np.pi*f
                    co[s,j], si[s,j] = math.cos(2*np.pi*f/sr), math.sin(2*np.pi*f/sr)
                    t60 = np.interp(math.log(f),knots,times[s])
                    decay[s,j] = math.exp(-math.log(1000)/(t60*sr)) if f < sr*.45 else 0.
                    # Release-front energy loses high partials much faster than
                    # the stored low-frequency displacement.
                    fast_t60 = max(.025,attack_t60*max(f/400,1.)**(-attack_slope))
                    fast_decay[s,j] = math.exp(-math.log(1000)/(fast_t60*sr)) if f < sr*.45 else 0.
            bridge = 0.
            age = n-excitation_start[s]
            if age < contact_samples:
                # Unit-area half-sine release: a flesh contact lasts several
                # milliseconds, unlike a near-impulsive pick release.
                weight = math.sin(np.pi*(age+.5)/contact_samples)*math.sin(np.pi/(2*contact_samples))
                for j in range(modes):
                    q[s,j] += excitation[s,j]*weight
            if age < 24:
                # A 0.5-ms smooth release front avoids manufacturing broadband
                # speech from a displacement step; the contact burst owns it.
                fast_weight = math.sin(np.pi*(age+.5)/24)*math.sin(np.pi/48)
                for j in range(modes):
                    fast_q[s,j] += fast_excitation[s,j]*fast_weight
            for j in range(modes):
                w = omega[s,j]
                a = q[s,j]*co[s,j]+v[s,j]/w*si[s,j]
                v[s,j] = (v[s,j]*co[s,j]-q[s,j]*w*si[s,j])*decay[s,j]*loss[s]
                q[s,j] = a*decay[s,j]*loss[s]
                fast_a = fast_q[s,j]*co[s,j]+fast_v[s,j]/w*si[s,j]
                fast_v[s,j] = (fast_v[s,j]*co[s,j]-fast_q[s,j]*w*si[s,j])*fast_decay[s,j]*loss[s]
                fast_q[s,j] = fast_a*fast_decay[s,j]*loss[s]
                if abs(fast_q[s,j]) < 1e-18 and abs(fast_v[s,j]) < 1e-15:
                    fast_q[s,j],fast_v[s,j] = 0.,0.
                if abs(q[s,j]) < 1e-18 and abs(v[s,j]) < 1e-15:
                    q[s,j],v[s,j] = 0.,0.
            if colliding[s]:
                displacement, velocity = np.dot(q[s],contact), np.dot(v[s],contact)
                if displacement < -gap:
                    # Mass-normalized modes: minimize potential energy in the
                    # position projection, kinetic energy in the velocity impulse.
                    position_norm = 0.
                    for j in range(modes):
                        position_norm += (contact[j]/omega[s,j])**2
                    correction = (-gap-displacement)/position_norm
                    for j in range(modes):
                        q[s,j] += correction*contact[j]/omega[s,j]**2
                    if velocity < 0:
                        dv = -(1+restitution)*velocity/norm
                        for j in range(modes):
                            v[s,j] += dv*contact[j]
                        impulse[owner[s],n] += impact_gain*(-velocity)
            for j in range(modes):
                bridge += (j+1)*(q[s,j]+fast_q[s,j])
            output[owner[s],n] += bridge*gains[s]
    return output, impulse


def render(events, seconds, owners=1, config=None):
    """11-column rows: sample, kind, string(0..3), owner, pitch, velocity,
    ramp samples, shape(0 linear/1 ease), position(0 preset), cents, token.
    Kinds 0 pizz, 1 release, 2 slide, 3 slap, 4 dead, 5 ghost, 6 damp,
    7 damped slap-back.
    Token-addressed releases cannot damp a later note on the same string.
    """
    c = config or load_config()
    if not math.isfinite(seconds) or seconds < 0 or type(owners) is not int or owners < 1:
        raise ValueError('invalid render duration/owners')
    rows = np.asarray(events,float).reshape(-1,11)
    if not np.all(np.isfinite(rows)):
        raise ValueError('events must be finite')
    for r in rows:
        if r[0] < 0 or r[1] not in range(8) or r[2] not in range(4) or r[3] not in range(owners) or not 0 <= r[5] <= 127 or r[6] < 0 or r[7] not in (0,1) or r[10] < 0 or r[10] != round(r[10]) or not (r[8] == 0 or .02 <= r[8] <= .48) or abs(r[9]) > 50:
            raise ValueError('invalid bass event')
        if r[1] in (0,2,3,4,5,7):
            assign_string(r[4],c,int(r[2])+1)
    rows = rows[np.argsort(rows[:,0],kind='stable')].copy()
    e,s,x,b,f = (c[k] for k in ('engine','strings','excitation','body','fingerboard'))
    sr, modes = e['sample_rate'], s['modes']
    if round(seconds*sr) == 0:
        return np.zeros((owners,0))
    amplitudes, noises = np.zeros((len(rows),modes)), np.zeros((len(rows),1024))
    attack_amplitudes = np.zeros_like(amplitudes)
    harmonics = np.arange(1,modes+1)
    detunings = {}
    for i,r in enumerate(rows):
        token = int(r[10])
        if r[1] in (0,3,4,5,7):
            rng = np.random.default_rng(np.random.SeedSequence([e['seed'],token,int(r[2]),int(r[3])]))
            cents = r[9]+rng.uniform(-x['intonation_cents'],x['intonation_cents'])
            # Open strings are tuned, not finger-stopped.
            if r[4] == s['open_midi'][int(r[2])]:
                cents = r[9]
            detunings[token] = cents
            r[4] += cents/100
            p = (r[8] or x['position'])*rng.uniform(1-x['position_jitter'],1+x['position_jitter'])
            f0 = 440*2**((r[4]-69)/12)
            B = s['stiffness'][int(r[2])]*2**((r[4]-s['open_midi'][int(r[2])])/6)
            hz = f0*harmonics*np.sqrt((1+B*harmonics**2)/(1+B))
            amp = (r[5]/127)**x['velocity_exponent']
            d = x['displacement_m']*amp
            if r[1] in (3,7):
                d *= f['slap_displacement_multiplier']
            if r[1] == 5:
                d *= .28
            # Triangular displacement, finite flesh-contact bandwidth and skin compliance.
            coefficients = 2*d*np.sin(np.pi*harmonics*p)/(np.pi**2*p*(1-p)*harmonics**2)
            if r[1] == 0:
                # Brief elastic release front; it cannot add a low fundamental.
                gate = np.clip((hz-350)/150,0,1)
                hardness = (r[5]/100)**x.get('attack_velocity_power',0)
                attack_amplitudes[i] = coefficients*x.get('attack_gain',0)*hardness*gate/np.sqrt(1+(hz/6500)**4)
                attack_amplitudes[i,hz > sr*.45] = 0
            coefficients *= np.exp(-.5*(hz*x['contact_ms']/1000)**2)/np.sqrt(1+(hz/x['lowpass_hz'])**4)
            coefficients[hz > sr*.45] = 0
            amplitudes[i] = coefficients
            noise = lfilter([1,-1],[1,-.85],rng.normal(size=1024))*np.exp(-np.arange(1024)/(sr*.004))
            noise /= max(np.sqrt(np.sum(noise**2)),1e-12)
            noises[i] = noise*x['noise_gain']*amp*(3 if r[1] in (3,4,5,7) else 1)
            if r[1] == 0 and 'release_noise_gain' in x:
                # Contact-release force, highpassed then reflected from the
                # plucking point. The comb's delay follows the sounding length.
                burst = rng.normal(size=1024)
                burst = lfilter([1,-1],[1,-math.exp(-2*np.pi*500/sr)],burst)
                pole = math.exp(-2*np.pi*7000/sr)
                burst = lfilter([1-pole],[1,-pole],burst)
                burst *= np.exp(-np.arange(1024)/(sr*x.get('release_noise_ms',4)/1000))
                comb = max(1,round(p*sr/f0))
                reflected = burst.copy()
                if comb < len(burst): reflected[comb:] -= .7*burst[:-comb]
                reflected /= max(np.linalg.norm(reflected),1e-12)
                noises[i] = reflected*x['release_noise_gain']*amp*(r[5]/100)**x.get('attack_velocity_power',0)
        elif r[1] == 2:
            r[4] += detunings.get(token,0)/100 + r[9]/100
    string, impact = _strings(rows,amplitudes,noises,round(seconds*sr),owners,sr,
        np.array(s['open_midi'],float),np.array(s['stiffness']),np.log(s['decay_hz']),
        np.array(s['decay_t60']),np.array(s['gain']),s['release_t60'],s['dead_t60'],
        f['position'],f['clearance_m'],f['restitution'],f['impact_gain'],
        round(x.get('flesh_contact_ms',3.)*sr/1000),attack_amplitudes,
        x.get('attack_t60',.3),x.get('attack_loss_slope',.8))
    audio = b['direct']*string + impact
    drive = string+impact
    for hz,t60,weight in zip(b['frequency_hz'],b['t60_seconds'],b['weights']):
        radius = math.exp(-math.log(1000)/(sr*t60))
        angle = 2*np.pi*hz/sr
        modal_in = (1-radius)*abs(1-radius*np.exp(-2j*angle))
        audio += weight*lfilter([modal_in],[1,-2*radius*math.cos(angle),radius**2],drive,axis=1)
    # Remove displacement offset without attenuating E1's fundamental significantly.
    audio = lfilter([1,-1],[1,-math.exp(-2*np.pi*8/sr)],audio,axis=1)*e['output_gain']
    return _output_guard(audio,.85,.98)


def note(on, off, pitch, velocity, string, token, owner=0, kind='pizz', position=0, cents=0):
    if not 0 <= on < off or kind not in ('pizz','slap','slap-back','dead','ghost'):
        raise ValueError('note needs increasing times and an attack kind')
    return [[round(on*48000),KINDS[kind],string-1,owner,pitch,velocity,0,0,position,cents,token],
            [round(off*48000),1,string-1,owner,0,0,0,0,0,0,token]]
