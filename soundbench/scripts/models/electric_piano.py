"""Hammer-excited tine/tonebar modes through an asymmetric magnetic pickup.

The effective pickup flux is differentiated to voltage. Large tine excursions
sample its curvature, creating velocity-dependent bark without sample layers.
Two-times synthesis and polyphase decimation suppress nonlinear aliasing.
"""
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt
from .keys_common import SR, ROOT, load_config as _load, validate, curve, smooth, finish
from .piano import hammer_force
from soundbench.scripts.paths import preset_path

CONFIG = preset_path('electric-piano')
RATIOS = np.array([1., 1.0012, 6.267, 17.55])


def load_config(path=CONFIG):
    cfg = _load('electric-piano',path)
    if not 0 < cfg['tine']['t60_s'] <= 30 or not .005 <= cfg['damper']['release_s'] <= 1:
        raise ValueError('invalid tine decay or damper release')
    if not .1 <= cfg['pickup']['gap'] <= 5 or not .01 <= cfg['pickup']['offset'] <= 3:
        raise ValueError('invalid pickup geometry')
    if cfg['effect']['mode'] not in ('off','tremolo','stereo'):
        raise ValueError('effect mode is off, tremolo or stereo')
    return cfg


def modal_parameters(pitch, velocity, cfg):
    hz = 440*2**((pitch-69)/12)*2**(cfg['tine'].get('tuning_cents',0)/1200)
    f = hz*RATIOS.copy()
    # The tonebar's coupled companion beats gently against the tine's root.
    f[1] = hz*2**(cfg['tine']['beat_cents']/1200)
    decay = np.clip(cfg['tine']['t60_s']*2**(-(pitch-60)*cfg['tine']['decay_register']/12), .5, 30.)
    rates = np.log(1000)/decay*np.array([1.,.65, cfg['tine']['upper_loss'],cfg['tine']['upper_loss']*2])
    amp = np.array([1.,cfg['tine']['tonebar_gain'],cfg['tine']['bell_gain']*(velocity/127)**1.3,.015*(velocity/127)**2])
    amp *= (velocity/127)**cfg['hammer']['velocity_exponent']
    return f, rates, amp


def pickup_voltage(x, cfg, sr=SR*2):
    p = cfg['pickup']; gap=p['gap']; offset=p['offset']
    # Flux at a laterally offset pole tip; derivative is the induced EMF.
    flux = 1/np.sqrt(gap*gap+(offset-x*p['excursion'])**2)
    return np.diff(flux,prepend=flux[0])*sr/(2*np.pi*440)


def pedal_release(note, controls, seconds):
    off = note['off']; down = False
    events = sorted(e for e in controls if e[1]==64)
    for t,_,v in events:
        if t <= off: down=v>=64
    if not down: return off
    return next((t for t,_,v in events if t>off and v<64),seconds)


def render(notes, seconds, config=None, controls=()):
    cfg = config or load_config(); validate(notes,seconds,controls,28,100)
    sr=SR*2; n=round(seconds*sr); out=np.zeros(n)
    rng=np.random.default_rng(cfg['engine']['seed'])
    for note in sorted(notes,key=lambda x:(x['time'],x['pitch'],x['off'],x['velocity'])):
        start=round(note['time']*sr); release=pedal_release(note,controls,seconds)
        end=min(n,round((release+cfg['damper']['release_s']*12)*sr))
        t=np.arange(end-start)/sr
        f,rates,amp=modal_parameters(note['pitch'],note['velocity'],cfg)
        # Dissipative felt contact provides a velocity-dependent impulse spectrum.
        hc={'engine':{'sample_rate':sr},'hammer':cfg['hammer']}
        force=hammer_force(note['velocity'],hc)
        x=np.zeros(len(t))
        for hz,rate,a in zip(f,rates,amp):
            if hz >= sr*.4: continue
            transfer=np.sum(force*np.exp(-2j*np.pi*hz*np.arange(len(force))/sr))
            phase=np.angle(transfer)
            x+=a*abs(transfer)*np.sin(2*np.pi*hz*t+phase)*np.exp(-rate*t)
        x*=1-np.exp(-t/cfg['hammer']['attack_s'])
        damp=np.exp(-np.maximum(t-(release-note['time']),0)/cfg['damper']['release_s'])
        x*=damp
        voltage=pickup_voltage(x,cfg,sr)
        # Pitch compensation reflects pickup inductance/voicing, not note normalization.
        voltage*=440/f[0]
        out[start:end]+=voltage
        when=round(release*sr)
        count=min(round(.09*sr),n-when)
        if count>0:
            tt=np.arange(count)/sr
            thump=sosfilt(butter(2,220,fs=sr,output='sos'),rng.normal(size=count))
            out[when:when+count]+=cfg['damper']['thump_gain']*(note['velocity']/127)**.5*thump*np.exp(-tt/.018)
    drive=cfg['amp']['drive']
    out=np.tanh(out*drive)/drive
    out=resample_poly(out,1,2)[:round(seconds*SR)]
    out=sosfilt(butter(2,cfg['amp']['lowpass_hz'],fs=SR,output='sos'),out)
    e=cfg['effect']; default={'off':0,'tremolo':1,'stereo':2}[e['mode']]
    mode=np.rint(curve(controls,92,len(out),default*127/2)*2/127).astype(int)
    depth=smooth(curve(controls,93,len(out),e['depth']*127)/127,.02)
    rate=smooth(.5+curve(controls,94,len(out),(e['rate_hz']-.5)/9.5*127)/127*9.5,.02)
    phase=np.cumsum(rate)*2*np.pi/SR
    result=np.column_stack((out,out))
    for ch in range(2):
        lfo=np.sin(phase+np.where(mode==2,ch*np.pi,0))
        result[:,ch]*=np.where(mode==0,1,1-depth*.5+depth*.5*lfo)
    result*=smooth(curve(controls,11,len(out),127)/127,.01)[:,None]
    return finish(result,cfg['engine']['output_gain'])
