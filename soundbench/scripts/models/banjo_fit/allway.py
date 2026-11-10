#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.2", "optax>=0.2,<0.3", "numpy>=2", "scipy>=1.14", "numba>=0.61", "soundfile>=0.13", "matplotlib>=3.9"]
# ///
"""Fit and validate the banjo with explicit held-out source performances."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('JAX_PLATFORMS','cpu')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from soundbench.scripts.models.banjo import load_config, CONFIG
from soundbench.scripts.models.banjo_fit import twin, chain, audio, allway_loss

FLAGS=dict(deployable=True,two_stage=True,glide=False,noise=False,room=True,mic=True,background=True,codec=True)
PER_NOTE=('width','corner','position','level','detune','onset','glide','glide_time','noise','noise_time','tilt','second_mix')
ACTIVE={'loss','mode_frequency','mode_time','mode_gain','width','corner','position','stages','second_mix','second_rate','second_detune','chain_room','chain_t60','chain_mic','chain_background','chain_codec','chain_codec_rolloff'}


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def transfer(p,train,heldout):
    """Share string latents without learning any held-out excitation parameters."""
    by_string={}
    for index,note in enumerate(train):
        by_string.setdefault(note['string'],[]).append(index)
    missing=sorted({note['string'] for note in heldout if note['string'] not in by_string})
    if missing:
        raise ValueError(f'held-out strings have no training support: {missing}')
    result=dict(p)
    for key in PER_NOTE:
        result[key]=jnp.array([jnp.mean(p[key][jnp.array(by_string[h['string']])]) for h in heldout])
    return result


def load_splits(directory,seconds):
    banks={}
    for name in ('train','development','test_notes','test_dynamics'):
        target,notes=audio.load(directory/(name+'.json'),seconds,native=True)
        banks[name]=(target,notes)
    # The shared capture PSD is estimated exclusively from training performances.
    shared=np.median([n['pre_onset_psd'] for n in banks['train'][1]],axis=0).tolist()
    for _,notes in banks.values():
        for note in notes:
            note['background_psd']=shared
    return banks


def physical_record(p,base,config):
    q=twin.physical(p,base,config); c=chain.physical(p)
    return {k:np.asarray(v).tolist() for k,v in dict(**q,**c).items()}


def fit(args):
    if args.out.exists():
        raise ValueError('output directory must be fresh')
    args.out.mkdir(parents=True)
    config=load_config(CONFIG)
    config['excitation']['fractional_pluck']=True
    from soundbench.scripts.models.banjo_fit.report import export_config
    # A TOML source records the actual configuration, including optional booleans.
    config_text='\n'.join('\n['+table+']\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in values.items()) for table,values in config.items())+'\n'
    (args.out/'config.toml').write_text(config_text)
    banks=load_splits(args.data,args.seconds)
    target,notes=banks['train']
    for name,(y,ns) in banks.items():
        np.savez_compressed(args.out/(name+'.npz'),audio=y,notes=json.dumps(ns))
    np.savez_compressed(args.out/'aligned.npz',audio=target)
    write_json(args.out/'notes.json',notes)
    p,base=twin.initial_parameters(config,notes,40)
    ancestry=[]
    if args.seed:
        old=json.loads(args.seed.read_text())
        if any(np.shape(old[k])!=np.shape(v) for k,v in p.items()):
            raise ValueError('seed shape differs from training instrument')
        p={k:jnp.array(old[k]) for k in p}
        ancestry=[dict(path=str(args.seed.resolve()),sha256=hashlib.sha256(args.seed.read_bytes()).hexdigest())]
    p=twin.reseed_bounded_latents(p,base)
    p=twin.canonical_excitation(p,base,config,notes)
    if not args.seed:
        p['second_mix']=jnp.full(len(notes),np.log(.3/.7))
        p['second_detune']=jnp.full(5,np.arctanh(3/8))
    # All starts reset the capture model to the exact identity.
    p.update(chain.initial_parameters(zero_start=True))
    if args.start:
        rng=np.random.default_rng(args.start)
        for key in ('position','stages','mode_frequency','second_detune'):
            p[key]=p[key]+jnp.array(rng.normal(0,.15,np.shape(p[key])))
        p=twin.canonical_excitation(p,base,config,notes)
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    for source in (ROOT/'scripts/models/banjo.py',ROOT/'scripts/models/string_loss.py'):
        sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    working_diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)
    (args.out/'working.diff').write_bytes(working_diff)
    lineage=dict(start='bounded seed instrument; identity recording chain' if args.seed else 'hand instrument; identity recording chain',
                 resume_chain=ancestry,argv=sys.argv,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
                 working_diff_sha256=hashlib.sha256(working_diff).hexdigest(),
                 config=str((args.out/'config.toml').resolve()),config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
                 source_sha256=sources,
                 flags=FLAGS,versions=dict(jax=jax.__version__,numpy=np.__version__,optax=optax.__version__),
                 split_source_hashes={name:[n['native_sha256'] for n in ns] for name,(_,ns) in banks.items()},
                 selection='minimum training objective; development compares methods; final test is not used to choose iterates')
    write_json(args.out/'lineage.json',lineage)
    prepared=allway_loss.prepare(jnp.array(target),notes)
    def objective(parameters):
        y=twin.render(parameters,base,config,notes,target.shape[-1],**FLAGS)
        value=allway_loss.objective(y,prepared)
        prior=.003*jnp.mean(jnp.exp(jnp.clip(parameters['mode_gain'],-15,6))/jnp.array(base[2]))
        prior+=.002*jnp.mean(parameters['mode_frequency']**2)
        return value+prior+chain.penalty(parameters,**FLAGS)
    optimizer=optax.chain(optax.clip_by_global_norm(10.),optax.adam(args.rate))
    state=optimizer.init(p)
    @jax.jit
    def step(parameters,state):
        value,gradient=jax.value_and_grad(objective)(parameters)
        gradient={k:v if k in ACTIVE else jnp.zeros_like(v) for k,v in gradient.items()}
        updates,state=optimizer.update(gradient,state,parameters)
        return optax.apply_updates(parameters,updates),state,value
    snapshots=args.out/'snapshots'; snapshots.mkdir()
    history=[]; traces=[]; best=None; best_value=np.inf; selected=0
    start=time.monotonic()
    # Each state is evaluated exactly once, before its update; final gets its own evaluation.
    for index in range(args.steps+1):
        if index<args.steps:
            updated,state,value=step(p,state)
        else:
            value=objective(p)
        number=float(value)
        if not np.isfinite(number):
            raise FloatingPointError(f'nonfinite objective at state {index}')
        history.append(number)
        if number<best_value:
            best={k:np.array(v) for k,v in p.items()}; best_value=number; selected=index
        if index%5==0 or index in (0,25,100,args.steps):
            np.savez_compressed(snapshots/f's{index:05d}.npz',step=index,objective=number,**{k:np.asarray(v) for k,v in p.items()})
            traces.append(dict(step=index,objective=number,physical=physical_record(p,base,config)))
            write_json(args.out/'loss.json',history)
            write_json(args.out/'trajectories.json',traces)
        if index%25==0 or index==args.steps:
            print(f'{index}/{args.steps} objective={number:.6f} elapsed={time.monotonic()-start:.1f}s',flush=True)
        if index<args.steps:
            p=updated
    np.savez_compressed(snapshots/f's{selected:05d}.npz',step=selected,objective=best_value,**best)
    p={k:jnp.array(v) for k,v in best.items()}
    write_json(args.out/'parameters.json',{k:v.tolist() for k,v in best.items()})
    write_json(args.out/'metrics.json',dict(steps=args.steps,selected_step=selected,flags=FLAGS,seconds=args.seconds,
               objective=best_value,physical={k:np.asarray(v).tolist() for k,v in twin.physical(p,base,config).items()},
               optimizer='Adam in dimensionless latent coordinates, global norm clip 10',learning_rate=args.rate))
    import gc
    jax.clear_caches(); gc.collect()
    # An engine-compatible dry export retains the fractional comb configuration.
    q={k:np.asarray(v) for k,v in twin.physical(p,base,config).items()}
    export_config(args.out/'export.toml',config,q,notes,two_stage=True,deployable=True)
    evaluate(args.out)


def evaluate(directory, *, include_test=False):
    config=load_config(directory/'config.toml')
    p={k:jnp.array(v) for k,v in json.loads((directory/'parameters.json').read_text()).items()}
    notes=json.loads((directory/'notes.json').read_text())
    _,base=twin.initial_parameters(config,notes,40)
    results={}
    from soundbench.scripts.models.banjo_fit import loss
    splits=('train','development','test_notes','test_dynamics') if include_test else ('train','development')
    for name in splits:
        with np.load(directory/(name+'.npz')) as z:
            target=z['audio']; local=json.loads(str(z['notes']))
        parameters=transfer(p,notes,local)
        dry_flags=dict(FLAGS,room=False,mic=False,background=False,codec=False)
        dry=np.asarray(twin.render(parameters,base,config,local,target.shape[-1],**dry_flags))
        prepared=allway_loss.prepare(jnp.array(target),local)
        draw_scores=[]
        prediction=None
        for draw in (1,2,3,4):
            candidate=np.asarray(chain.apply(jnp.array(dry),parameters,local,room=True,mic=True,background=True,codec=True,realization=draw))
            draw_scores.append(float(allway_loss.objective(jnp.array(candidate),prepared)))
            if draw==1:
                prediction=candidate
        objective=float(np.mean(draw_scores))
        spectral,envelope=loss.terms(jnp.array(prediction),loss.prepare(jnp.array(target),local))
        diagnostic=allway_loss.noise_units(target,prediction,local)
        np.savez_compressed(directory/(name+'-predictions.npz'),target=target,prediction=prediction,dry=dry)
        # Module owns diagnostic structure; large residual maps remain numeric companions.
        arrays={k:v for k,v in diagnostic.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(directory/(name+'-residuals.npz'),**arrays)
        summary={k:v for k,v in diagnostic.items() if not isinstance(v,np.ndarray)}
        results[name]=dict(notes=len(local),noise_scaled_objective=objective,nuisance_draw_scores=draw_scores,
                           nuisance_draw_std=float(np.std(draw_scores)),
                           conditioning='measured onset, fundamental frequency, full-probe RMS and assumed string assignment',
                           legacy_spectral=float(spectral),legacy_envelope=float(envelope),diagnostic=summary)
        print(name,objective,flush=True)
        jax.clear_caches()
    write_json(directory/'scores.json',results)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'build/banjo-allway/data')
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--rate',type=float,default=.01)
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--seed',type=Path)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--test',action='store_true',help='Evaluate final held-out pitches and alternate dynamics after method selection')
    args=parser.parse_args()
    if args.evaluate: evaluate(args.out,include_test=args.test)
    else: fit(args)
