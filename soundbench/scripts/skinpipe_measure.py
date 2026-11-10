#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "numba", "mido", "soundfile"]
# ///
"""Audio-domain measurements shared by acceptance tests and the probe lab."""
import copy
import json
import time
import numpy as np
from scipy.signal import find_peaks
from soundbench.scripts.models.skinpipe import render, load_config, frequency
from soundbench.scripts.winds_diagnose import pitch

SR = 48000

def note(p=62, time=0, off=4, velocity=90):
    return dict(pitch=p,time=time,off=off,velocity=velocity)

def cc(control,value,time=0):
    return dict(control=control,value=value,time=time)

def rms(y): return float(np.sqrt(np.mean(np.asarray(y)**2)))
def db_ratio(a,b): return float(20*np.log10(max(a,1e-20)/max(b,1e-20)))
def span(y,a,b): return y[...,round(a*SR):round(b*SR)]

def clean():
    c=load_config()
    c['wobble_cents']=c['squawk_cents']=c['sag_cents']=0
    return c

def held(p,config=None,damper=False,details=True):
    return render([note(p,off=4.5)],12,config or load_config(),
                  [cc(24,0,4),cc(64,0 if damper else 127)],details=details)

def cents(y,p,c,a=1.5,b=3.5):
    target=frequency(p,c)
    actual,_=pitch(span(y,a,b),SR,target)
    return float(1200*np.log2(actual/target))

def decay_time(y,a=4.5,b=9):
    times=np.arange(a,b,.1)
    levels=np.array([20*np.log10(max(rms(span(y,t,t+.1)),1e-20)) for t in times])
    slope,intercept=np.polyfit(times,levels,1)
    return float(-60/slope), times, levels

def bloom_metrics(c=None):
    c=c or load_config()
    d,dp=held(62,c);e,ep=held(64,c)
    bloom=db_ratio(rms(span(dp['courses'],1.5,4)),rms(span(ep['courses'],1.5,4)))
    halo=db_ratio(rms(span(d,4.9,5.1)),rms(span(d,3.5,3.9)))
    deficit=db_ratio(rms(span(d,4.9,5.1)),rms(span(e,4.9,5.1)))
    t60,_,_=decay_time(d)
    return dict(bloom_db=bloom,halo_at_1s_db=halo,e4_halo_deficit_db=deficit,t60_s=t60,
                course_bloom_db={str(p):db_ratio(rms(span(dp['courses'][i],1.5,4)),rms(span(ep['courses'][i],1.5,4)))
                                 for i,p in enumerate(c['courses'])},
                skin_under_dry_db=db_ratio(rms(span(dp['skin'],1.5,4)),rms(span(dp['dry'],1.5,4)))),(d,dp,e,ep)

def capture_curve():
    """Vary bare bore length, never string tuning or pitch-selected drive."""
    c=load_config(); result=[]
    for detune in range(-20,21,2):
        config=copy.deepcopy(c)
        config['bore_correction'][0]-=detune
        free=render([note(62,off=5)],5.2,config,[cc(64,127)])
        damped=render([note(62,off=5)],5.2,config,[cc(64,0)])
        result.append(dict(bore_detune_cents=detune,free_cents=cents(free,62,c,3,5),
                           damped_cents=cents(damped,62,c,3,5)))
    return result


def measure_all(stability_seconds=4.5):
    c=load_config()
    metrics,_=bloom_metrics(c)
    # Preset squawk and seeded pressure wander remain present in pitch acceptance.
    errors={}
    for p in range(57,72):
        y=render([note(p)],4.5,c)
        errors[str(p)]=cents(y,p,c)
    metrics['pitch_cents']=errors
    # Hold the same stochastic trajectory for the free/damped A/B.
    lock={}
    for p in sorted({q+d for q in c['courses'] for d in (-1,1) if 57<=q+d<=71}):
        free=render([note(p)],4.5,c,[cc(64,127)])
        damped=render([note(p)],4.5,c,[cc(64,0)])
        lock[str(p)]=cents(free,p,c)-cents(damped,p,c)
    metrics['lock_in_cents']=lock
    pressure_energy={}
    for value in sorted({85,112,*range(90,109)}):
        _,parts=render([note(62,off=4.5)],6,c,[cc(2,value),cc(24,0,4)],details=True)
        pressure_energy[str(value)]=rms(span(parts['courses'],1.5,4))
    metrics['pressure_course_energy_db_relative_cc96']={p:db_ratio(v,pressure_energy['96'])
                                                       for p,v in pressure_energy.items()}
    coupled={}
    isolated=copy.deepcopy(c);isolated['chamber_feedback']=0
    for p in range(57,72):
        a=render([note(p)],4.5,c);b=render([note(p)],4.5,isolated)
        coupled[str(p)]=cents(a,p,c)-cents(b,p,c)
    metrics['feedback_shift_cents']=coupled
    y=render([note(62,off=4.5)],9,c,[cc(24,0,4),cc(64,0,6)])
    before=rms(span(y,5.8,6))
    times=np.arange(6,6.4,.01)
    levels=np.array([db_ratio(rms(span(y,t,t+.02)),before) for t in times])
    below=np.flatnonzero(levels<=-40)
    metrics['damper_40db_s']=float(times[below[0]]-6) if len(below) else None
    metrics['damper_at_300ms_db']=db_ratio(rms(span(y,6.3,6.32)),before)
    metrics['damper_click_peak_ratio']=float(max(abs(np.diff(span(y,6,6.3))))/max(abs(span(y,5.8,6))))
    pluck_errors={}
    for p in c['courses']:
        y=render([],5,c,plucks=[note(p,off=.1)])
        pluck_errors[str(p)]=cents(y,p,c,.3,1)
    metrics['pluck_cents']=pluck_errors
    d,_=held(62,c)
    x=span(d,4.9,5.4);power=abs(np.fft.rfft(x*np.hanning(len(x))))**2
    f=np.fft.rfftfreq(len(x),1/SR);mask=np.zeros(len(f),bool)
    for p in c['courses']:
        f0=frequency(p,c)
        for harmonic in range(1,int(24000/f0)+1): mask |= abs(f-harmonic*f0)<max(4,harmonic*f0*.003)
    metrics['halo_course_harmonic_power_share']=float(power[mask].sum()/power.sum())
    worst,case=0,None
    start=time.perf_counter()
    for p in range(57,72):
        for pressure in (70,96,112,127):
            for lean in (0,127):
                for damp in (0,127):
                    y=render([note(p,off=stability_seconds)],stability_seconds+.7,c,
                             [cc(2,pressure),cc(74,lean),cc(64,damp)])
                    peak=float(max(abs(y)))
                    if peak>worst: worst,case=peak,[p,pressure,lean,damp]
        print(f'stability MIDI {p}: worst {worst:.4f}',flush=True)
    metrics['worst_peak']=worst;metrics['worst_case_pitch_cc2_cc74_cc64']=case
    metrics['stability_cases']=240;metrics['stability_seconds_per_case']=stability_seconds+.7
    metrics['stability_sweep_wall_s']=time.perf_counter()-start
    start=time.perf_counter()
    y=render([note(62,off=171)],180,c,[cc(24,0,171)])
    metrics['render_180s_wall_s']=time.perf_counter()-start
    import hashlib
    import platform
    from soundbench.scripts.models.skinpipe import ROOT, CONFIG
    metrics['capture_curve']=capture_curve()
    metrics['parameters_sha256']=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    metrics['engine_sha256']=hashlib.sha256((ROOT/'scripts/models/skinpipe.py').read_bytes()).hexdigest()
    metrics['dependencies_sha256']={name:hashlib.sha256((ROOT/f'scripts/models/{name}.py').read_bytes()).hexdigest()
                                    for name in ('banjo','mallorca_winds')}
    metrics['benchmark_machine']=platform.machine()
    return metrics

if __name__=='__main__':
    from soundbench.scripts.models.skinpipe import ROOT
    metrics=measure_all()
    out=ROOT/'build/skinpipe-probe';out.mkdir(parents=True,exist_ok=True)
    (out/'acceptance.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
