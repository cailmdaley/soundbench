"""Passive reciprocal bridge admittance for two/three rotating-wave strings."""
import numpy as np


def bridge_matrix(offsets, intrinsic, admittance):
    """dE/dt = -2 r ||z||² - 2 g |bᵀz|² <= 0 for r,g >= 0.

    The bridge velocity is bᵀz. Its reaction -g b bᵀ z couples every string
    reciprocally. This is a normalized real admittance; units are loss / second.
    """
    n=len(offsets); b=np.ones(n)/np.sqrt(n)
    return 1j*np.diag(offsets)-intrinsic*np.eye(n)-admittance*np.outer(b,b)


def coupled_poles(hz,count,cents,late_rate,prompt_rate):
    if count==1:
        return np.array([-late_rate,-prompt_rate],complex), {'admittance':0,'intrinsic':late_rate,'slow':0,'fast':1}
    offsets=2*np.pi*hz*(2**((np.arange(count)-(count-1)/2)*cents/1200)-1)
    g=max(prompt_rate-late_rate,.2)
    reactive=0.0
    for _ in range(80):
        vals,vectors=np.linalg.eig(bridge_matrix(offsets,0,g+1j*reactive))
        slow=int(np.argmax(vals.real)); fast=int(np.argmin(vals.real))
        if -vals[slow].real<late_rate*.8: break
        # Reactive board loading can separate bright/dark frequencies without
        # forcing an unrealistically short bright-mode decay. Bound its shift.
        if reactive<2*np.pi*hz*.001:
            reactive=min(2*np.pi*hz*.001,max(.2,reactive*1.5))
        else:
            g*=1.5
    intrinsic=max(.005,late_rate+vals[slow].real)
    vals-=intrinsic
    b=np.ones(count)/np.sqrt(count)
    return vals, {'admittance':g,'reactance':reactive,'intrinsic':intrinsic,'slow':slow,'fast':fast,'vectors':vectors,'radiation':b@vectors,'offsets':offsets}


def initial_state(info,residues):
    """Recover string displacement/velocity amplitudes from fitted radiation.

    Residues set the measured excitation state, not independent decay rates.
    Evolution remains exactly the coupled bridge matrix's eigensolution.
    """
    radiation = info['radiation']
    if np.any((abs(radiation)<1e-12) & (abs(residues)>1e-12)):
        raise ValueError('nonzero radiation assigned to an unobservable mode')
    modal = np.divide(residues, radiation, out=np.zeros_like(residues), where=abs(radiation)>=1e-12)
    return info['vectors']@modal


def weight_initial_excitation(info, residues, weights):
    """Redistribute excitation over strings without changing passive dynamics.

    Nonnegative string weights perturb the inferred complex initial state.
    Its total energy is preserved, and projection excites every observable
    eigenmode, including the third unison component. The bridge matrix and its
    reciprocal damping remain unchanged.
    """
    weights = np.asarray(weights, float)
    state = initial_state(info, residues)
    weighted = state*weights
    original_energy = float(np.vdot(state, state).real)
    energy = float(np.vdot(weighted, weighted).real)
    if energy <= 1e-30:
        return np.zeros_like(residues)
    weighted *= np.sqrt(original_energy/energy)
    modal = np.linalg.solve(info['vectors'], weighted)
    return info['radiation']*modal
