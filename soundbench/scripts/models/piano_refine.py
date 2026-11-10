#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "soundfile"]
# ///
"""Fit passive unison bridge dynamics and sample-free residual radiation.

Residuals subtract narrowband, time-varying sinusoidal partial reconstructions.
They are observations for a stochastic body filter bank, never runtime audio.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import json
import numpy as np
from scipy.signal import find_peaks, stft, butter, sosfreqz, sosfiltfilt
from scipy.optimize import nnls
if __package__:
    from .piano_fit import ROOT, LAYERS, MODES, source_map, load
else:
    from piano_fit import ROOT, LAYERS, MODES, source_map, load

TIMES = np.array([0, .01, .025, .05, .1, .2, .4, .7, 1, 1.5, 2, 3, 4, 5, 6, 8])
EDGES = np.geomspace(25, 11000, 25)


def mistuning(y, sr, f0, B):
    x=y[int(.3*sr):].mean(axis=1)
    nfft=262144
    mag=abs(np.fft.rfft(x*np.hanning(len(x)),nfft))
    freq=np.fft.rfftfreq(nfft,1/sr)
    peaks,_=find_peaks(mag,distance=max(1,round(.2*nfft/sr)))
    spacings=[]
    for k in range(1,13):
        hz=f0*k*np.sqrt(1+B*k*k)
        if hz>9000: continue
        band=peaks[abs(freq[peaks]-hz)<hz*.006]
        if len(band)<2: continue
        band=band[mag[band]>mag[band].max()*.25]
        if len(band)<2: continue
        strongest=sorted(band,key=lambda p:mag[p],reverse=True)[:3]
        fs=np.sort(freq[strongest])
        cents=1200*np.log2(fs[1:]/fs[:-1])
        spacings.extend(cents[(cents>.05)&(cents<5)].tolist())
    return float(np.clip(np.median(spacings) if spacings else 1.2,.1,3)),len(spacings)


def sinusoidal_residual(y,sr,f0,B):
    """Subtract fitted carriers with slowly varying complex amplitudes.

    A narrow positive-frequency band estimates each carrier's demodulated
    envelope. Zero padding keeps the FFT's periodic seam outside the note.
    The sum of reconstructed carrier bands is the sinusoidal estimate.
    """
    pad=sr
    padded=np.pad(y,((pad,pad),(0,0)))
    spectrum=np.fft.rfft(padded,axis=0)
    f=np.fft.rfftfreq(len(padded),1/sr)
    harmonic=np.zeros(len(f))
    magnitude=np.sqrt(np.mean(abs(spectrum)**2,axis=1))
    peaks,_=find_peaks(magnitude)
    for k in range(1,MODES+1):
        hz=f0*k*np.sqrt(1+B*k*k)
        if hz>10500: continue
        candidates=peaks[abs(f[peaks]-hz)<max(12,f0*.18)]
        if not len(candidates): continue
        strongest=magnitude[candidates].max()
        if strongest<magnitude.max()*.0005: continue
        candidates=candidates[magnitude[candidates]>strongest*.1]
        for p in candidates:
            width=max(3.0,hz*.001)
            distance=abs(f-f[p])
            weight=np.clip((width*2-distance)/width,0,1)
            weight=.5-.5*np.cos(np.pi*weight)
            harmonic=np.maximum(harmonic,weight)
    estimate=np.fft.irfft(spectrum*harmonic[:,None],n=len(padded),axis=0)[pad:pad+len(y)]
    return y-estimate,estimate


def fit_knock(residual,sr):
    n=round(.2*sr); t=np.arange(n)/sr
    low=sosfiltfilt(butter(2,(40,100),fs=sr,btype='bandpass',output='sos'),residual,axis=0)[:n].mean(axis=1)
    best=None
    for hz in (40,55,70,85,100):
        for tau in (.02,.035,.06,.1,.15):
            envelope=np.exp(-t/tau)*np.minimum(t/.003,1)
            basis=np.column_stack((np.sin(2*np.pi*hz*t),np.cos(2*np.pi*hz*t)))*envelope[:,None]
            gain=np.linalg.lstsq(basis,low,rcond=None)[0]
            error=np.mean((low-basis@gain)**2)
            if best is None or error<best[0]: best=(error,hz,tau,gain)
    _,hz,tau,gain=best
    t=np.arange(len(residual))/sr
    signal=(gain[0]*np.sin(2*np.pi*hz*t)+gain[1]*np.cos(2*np.pi*hz*t))*np.exp(-t/tau)*np.minimum(t/.003,1)
    return [hz,tau,*gain.tolist()],residual-signal[:,None]


@lru_cache(maxsize=64)
def bank_matrix(frequencies):
    fs=48000
    f=np.linspace(0,fs/2,65536,endpoint=False)
    transfer=np.ones(len(f))
    for hz in frequencies:
        if hz>10500: continue
        width=max(3,hz*.0025)
        weight=np.clip((2*width-abs(f-hz))/width,0,1)
        transfer*=.5+.5*np.cos(np.pi*weight)
    matrix=[]
    for low,high in zip(EDGES[:-1],EDGES[1:]):
        sos=butter(2,(low,high),fs=fs,btype='bandpass',output='sos')
        _,h=sosfreqz(sos,worN=65536)
        energy=abs(h)**2
        matrix.append([np.sum((energy*transfer**2)[(f>=lo)&(f<hi)])/np.sum(energy) for lo,hi in zip(EDGES[:-1],EDGES[1:])])
    return np.array(matrix).T


def calibrate_bank(levels,frequencies):
    matrix=bank_matrix(tuple(frequencies))
    result=[]
    for target in levels.T**2:
        scale=max(target.max(),1e-20)
        weights,_=nnls(matrix,target/scale,maxiter=500)
        result.append(np.sqrt(weights*scale))
    return np.array(result).T


def residual_profile(residual,sr):
    f,t,z=stft(residual,fs=sr,nperseg=2048,noverlap=1536,axis=0,boundary='zeros')
    energy=np.mean(abs(z)**2,axis=1)/1.5
    levels=[]
    for low,high in zip(EDGES[:-1],EDGES[1:]):
        rms=np.sqrt(2*np.sum(energy[(f>=low)&(f<high)],axis=0))
        # Piecewise RMS envelope, not a sampled waveform or a learned neural IR.
        values=[]
        for center in TIMES:
            near=abs(t-center)<=max(.025,center*.08)
            value=float(np.sqrt(np.mean(rms[near]**2))) if np.any(near) else float(rms[-1])
            values.append(value)
        # The final tail is a finite stochastic radiation approximation.
        values[-2]=values[-3]*np.exp(-1/1.5)
        values[-1]=values[-3]*np.exp(-3/1.5)
        levels.append(values)
    return np.array(levels)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body-only',action='store_true')
    args=parser.parse_args()
    base=json.loads((ROOT/'instruments/models/piano-modal.json').read_text())
    observations=np.load(ROOT/'build/piano-fit/observations.npz')
    obs=observations['observed']; t=observations['times']
    rows=source_map(); profiles=[]; bodies=[]; diagnostics=[]
    tuning=json.loads((ROOT/'references/piano/fit.json').read_text())['tuning']
    for profile in base['profiles']:
        pitch=profile['pitch']; indices=[i for i,r in enumerate(rows) if r[0]==pitch]
        f0,B=tuning[str(pitch)]['f0'],tuning[str(pitch)]['B']
        y,sr=load(rows[indices[7]][2],seconds=8)
        cents,resolved=mistuning(y,sr,f0,B)
        count=1 if pitch<31 else 2 if pitch<43 else 3
        rates=[]; prompt=[]; amps=[]; gains=[]; body=[]; knocks=[]
        nominal=440*2**((pitch-69)/12)
        ks=np.arange(1,MODES+1)
        frequencies=nominal*ks*np.sqrt((1+B*ks*ks)/(1+B))
        for i in indices:
            layer=rows[i][1]-1
            measured=obs[i]
            late=(t>=2)&(t<=4.5)
            fit=np.polyfit(t[late],np.log(np.maximum(measured[late],1e-10)),1)
            target=np.clip(-fit[0],.025,15)
            # Weak/rising bands cannot identify a material loss rate.
            reliable=(measured.max(axis=0)>measured.max()*.005)&(-fit[0]>.025)
            oldslow=np.array(profile['slow_rate'][layer]); oldfast=np.array(profile['fast_rate'][layer])
            target=np.where(reliable,target,oldslow)
            # The late regression extrapolation isolates the aftersound level.
            slow_amp=np.exp(fit[1])
            slow_amp=np.minimum(slow_amp,measured.max(axis=0)*2)
            slow_amp=np.where(reliable,slow_amp,np.array(profile['slow_amp'][layer]))
            fast_amp=np.maximum(np.array(profile['fast_amp'][layer])+np.array(profile['slow_amp'][layer])-slow_amp,0)
            rates.append(target.tolist()); prompt.append(oldfast.tolist())
            amps.append(slow_amp.tolist()); gains.append(fast_amp.tolist())
            y,sr=load(rows[i][2],seconds=8)
            residual,partial=sinusoidal_residual(y,sr,f0,B)
            knock,noise=fit_knock(residual,sr)
            knocks.append(knock)
            body.append(calibrate_bank(residual_profile(noise,sr),frequencies).tolist())
            if layer in (1,4,7,9):
                diagnostics.append({'pitch':pitch,'layer':layer+1,'mistuning_cents':cents,'resolved_spacings':resolved,
                                    'reference_rms':float(np.std(y)),'residual_rms':float(np.std(residual)),
                                    'partial_rms':float(np.std(partial))})
        profiles.append({'pitch':pitch,'mistuning_cents':cents,'resolved_spacings':resolved,
                         'late_rate':rates,'prompt_rate':prompt,'slow_amp':amps,'prompt_amp':gains})
        bodies.append({'pitch':pitch,'rms':body,'knock':knocks})
        print('fit',pitch,'mistuning',round(cents,3),'spacings',resolved,flush=True)
    target=ROOT/'instruments/models'
    if not args.body_only:
        (target/'piano-coupled.json').write_text(json.dumps({'schema':1,'velocities':LAYERS,'profiles':profiles},separators=(',',':'))+'\n')
    (target/'piano-body.json').write_text(json.dumps({'schema':1,'source':base['source'],'velocities':LAYERS,'times':TIMES.tolist(),'band_edges':EDGES.tolist(),'profiles':bodies},separators=(',',':'))+'\n')
    (ROOT/'references/piano/residual-fit.json').write_text(json.dumps(diagnostics,indent=2)+'\n')

if __name__=='__main__': main()
