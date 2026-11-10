#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11", "soundfile>=0.12", "matplotlib>=3.8", "tomli-w>=1"]
# ///
"""Original bass probes and a generic isolated-note spectral fit.

No reference recording is downloaded. Supply a local WAV and JSON note list
for a private fit; audio and figures belong in an ignored output directory.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.optimize import differential_evolution
import tomli_w
from soundbench.scripts.paths import preset_names, preset_path
from soundbench.scripts.models.electric_bass import ROOT, SR, load_config, render, note, assign_string


def lufs(path):
    p=subprocess.run(['ffmpeg','-hide_banner','-nostats','-i',str(path),'-af','ebur128=peak=true','-f','null','-'],capture_output=True,text=True,check=True)
    return float(re.findall(r'I:\s+(-?\d+(?:\.\d+)?) LUFS',p.stderr)[-1])


def export(path,y,target=None,plot=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    wav=path.with_suffix('.wav'); sf.write(wav,y,SR,subtype='PCM_24')
    raw=lufs(wav)
    gain=10**(((target if target is not None else -23)-raw)/20)
    if np.max(abs(y),initial=0)*gain>.95:
        raise ValueError('static loudness matching would clip; choose a quieter target')
    y=y*gain;sf.write(wav,y,SR,subtype='PCM_24')
    mp3=path.with_suffix('.mp3')
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(wav),'-codec:a','libmp3lame','-b:a','192k',str(mp3)],check=True)
    if plot:
        subprocess.run([sys.executable,'-m','soundbench.scripts.spectrogram',str(mp3),'--out',str(path.with_suffix('.png')),'--title',path.stem],check=True,stdout=subprocess.DEVNULL)
    return dict(raw_lufs=raw,matched_lufs=lufs(mp3),gain_db=20*np.log10(gain),peak=float(np.max(abs(y),initial=0)))


def probes(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metrics={}
    for preset in preset_names('electric-bass'):
        c=load_config(preset_path('electric-bass',preset))
        sweeps=[]
        for i,p in enumerate(range(28,68)):
            sweeps+=note(i*.55,i*.55+.3,p,90,assign_string(p,c)+1,i)
        velocity=[]
        for i,v in enumerate(range(20,128,3)):
            velocity+=note(i*.45,i*.45+.28,40,v,3,i)
        gestures=[]
        for i,k in enumerate((0,0,7,6)):
            gestures+=note(i*1.2,i*1.2+.7,40,100,3,i,kind=k,hardness=int(i==1))
        gestures+=note(4.8,5.8,40,100,3,4)
        gestures.append([round(5.0*SR),5,2,0,0,0,0,0,0,0,4])
        gestures+=note(6.0,7.1,40,90,3,5)
        gestures+=note(6.35,7.1,43,90,3,6,kind=3)
        gestures+=note(6.7,7.1,40,90,3,7,kind=4)
        gestures+=note(7.4,8.5,40,95,3,8)
        gestures.append([round(7.65*SR),2,2,0,45,0,.4*SR,1,0,0,8])
        # Original eight-bar E/A/B phrase: syncopated sixteenths, alternating
        # short and held notes, dead strokes, and a one-bar turnaround.
        demo=[];token=0;beat=.75
        for bar in range(8):
            root=[28,28,33,28,28,33,35,28][bar]
            pattern=[(0,root,.45,104,0),(1.5,root+12,.20,85,0),(2.0,root,.13,70,6),(2.75,root+7,.15,90,0),(3.25,root+12,.36,100,0)]
            if bar==7:pattern=[(0,40,.35,100,0),(1,38,.25,90,0),(1.75,35,.16,85,0),(2.5,33,.2,88,0),(3,28,.65,110,0)]
            for pos,p,dur,v,k in pattern:
                on=(bar*4+pos)*beat
                demo+=note(on,on+dur,p,v,assign_string(p,c)+1,token,kind=k);token+=1
        for name,rows,seconds in [('register',sweeps,22.8),('velocity',velocity,17.5),('articulations',gestures,9.3),('demo',demo,25.)]:
            y=render(rows,seconds,config=c)[0]
            metrics[preset+'-'+name]=export(out/(preset+'-'+name),y)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')


def spectral(y):
    f,t,z=stft(y,SR,nperseg=4096,noverlap=3072,boundary='zeros')
    mask=(f>=35)&(f<=4000)
    return f[mask],t,abs(z[mask])


def spectral_distance(a,b):
    """RMS-normalized log magnitude error, including model-only ringing."""
    a=a/max(np.sqrt(np.mean(a*a)),1e-12)
    b=b/max(np.sqrt(np.mean(b*b)),1e-12)
    _,_,A=spectral(a);_,_,B=spectral(b)
    floor=max(np.max(A),np.max(B))*10**(-45/20)
    mask=(A>floor)|(B>floor)
    if not np.any(mask):
        return 0.
    return float(np.mean(abs(20*np.log10((B[mask]+floor)/(A[mask]+floor)))))


def align_reference(z):
    """Trim to a 3-ms RMS 20%-of-peak threshold, retaining 5ms pre-attack."""
    width=round(.003*SR)
    env=np.sqrt(np.convolve(z*z,np.ones(width)/width,'same')+1e-20)
    above=np.where(env>np.max(env)*.20)[0]
    trim=max(0,int(above[0])-round(.005*SR)) if len(above) else 0
    return z[trim:],trim/SR


def fit(reference,notes_path,out,iterations=8):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    y,sr=sf.read(reference)
    if y.ndim>1:y=y.mean(axis=1)
    if sr!=SR:
        from scipy.signal import resample_poly
        from math import gcd
        g=gcd(sr,SR);y=resample_poly(y,SR//g,sr//g)
    records=json.loads(Path(notes_path).read_text())
    base=load_config(); base['amp']['compressor_ratio']=1.; targets=[]
    for n in records:
        z=y[round(n['on']*SR):round(n['off']*SR)]
        # Centered low-frequency onset windows can precede actual attacks.
        # Align fit clips to a 3-ms RMS threshold, retaining 5ms pre-attack.
        aligned,shift=align_reference(z)
        n['requested_on']=n['on']; n['on']+=shift
        targets.append(aligned)
    def config(v):
        c=copy.deepcopy(base)
        c['excitation'].update(finger_ms=float(v[0]),lowpass_hz=float(v[1]),position=float(v[2]))
        c['pickup']['position']=[float(v[3])]
        c['amp'].update(cab_hz=float(v[4]),drive=float(v[5]),compressor_ratio=1.)
        for lane in range(4):c['strings']['decay_t60'][lane]=[float(t*v[6]) for t in base['strings']['decay_t60'][lane]]
        return c
    # Cache dry string renders across pickup/cab/drive proposals would make
    # this fitter faster. This bounded small study favors explicit simplicity.
    def synth(c,n):
        dur=n['off']-n['on'];p=n['pitch']+n.get('cents',0)/100
        return render(note(0,dur,p,90,assign_string(p,c)+1,0),dur,config=c)[0]
    distance=spectral_distance
    def objective(v):
        c=config(v)
        return np.mean([distance(z,synth(c,n)) for z,n in zip(targets[:3],records[:3])])
    bounds=[(.6,4.),(400,5000),(.08,.35),(.12,.36),(700,5000),(0,8),(.35,2.)]
    initial=[2.6,1600,.19,.24,2800,2,1]
    before=objective(initial)
    result=differential_evolution(objective,bounds,maxiter=iterations,popsize=4,seed=817,polish=False,x0=initial,workers=1)
    c=config(result.x)
    (out/'fit.toml').write_text(tomli_w.dumps(c))
    per=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i,(z,n) in enumerate(zip(targets,records)):
        model=synth(c,n); prior=synth(base,n)
        per.append(dict(note=n,split='train' if i<3 else 'holdout',before=distance(z,prior),after=distance(z,model)))
        sf.write(out/f'note-{i}-reference.wav',z,SR)
        sf.write(out/f'note-{i}-model.wav',model,SR)
        a=z/max(np.sqrt(np.mean(z*z)),1e-12);b=model/max(np.sqrt(np.mean(model*model)),1e-12)
        f,t,A=spectral(a);_,_,B=spectral(b)
        floor=max(np.max(A),np.max(B))*10**(-60/20)
        ref=20*np.log10(A+floor);mod=20*np.log10(B+floor)
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        for ax,data,title in zip(axes,[ref,mod,mod-ref],['Reference','Model','Model − reference (dB)']):
            im=ax.pcolormesh(t,f,data,shading='auto',cmap='coolwarm' if '−' in title else 'magma',vmin=-20 if '−' in title else np.max(ref)-60,vmax=20 if '−' in title else np.max(ref))
            ax.set_yscale('log');ax.set_ylim(35,4000);ax.set_title(title);ax.set_xlabel('seconds');fig.colorbar(im,ax=ax)
        fig.tight_layout();fig.savefig(out/f'note-{i}-triptych.png');plt.close(fig)
    metrics=dict(objective='reference-or-model-active-bin mean absolute dB error, RMS normalized, 35–4000 Hz, 4096 FFT / 1024 hop / −45 dB floor',before=float(before),after=float(result.fun),parameters=result.x.tolist(),evaluations=int(result.nfev),iterations=int(result.nit),notes=per)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('probes');a.add_argument('--out',default=str(ROOT/'build/electric-bass-probe'))
    a=s.add_parser('fit');a.add_argument('--reference',required=True);a.add_argument('--notes',required=True);a.add_argument('--out',required=True);a.add_argument('--iterations',type=int,default=8)
    a=p.parse_args()
    if a.command=='probes':probes(a.out)
    else:fit(a.reference,a.notes,a.out,a.iterations)
