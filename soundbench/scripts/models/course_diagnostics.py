"""Tonal course measurements shared by the mandolin tests and evidence plots."""
import numpy as np
from scipy import signal

SR=48000


def partial_envelopes(members,pitch,partial=1):
    hz=440*2**((pitch-69)/12)*partial
    sos=signal.butter(3,[hz-18,hz+18],btype='bandpass',fs=SR,output='sos')
    bands=signal.sosfiltfilt(sos,members,axis=1)
    analytic=signal.hilbert(bands,axis=1)
    envelopes=abs(analytic)
    interference=abs(analytic.sum(axis=0))/np.maximum(envelopes.sum(axis=0),1e-20)
    return envelopes,interference


def fundamental(y,pitch):
    f0=440*2**((pitch-69)/12)
    z=y[4800:43200]*np.hanning(38400)
    n=524288
    power=abs(np.fft.rfft(z,n))
    frequencies=np.fft.rfftfreq(n,1/SR)
    use=np.flatnonzero(abs(frequencies-f0)<f0*.01)
    k=use[np.argmax(power[use])]
    a,b,c=np.log(power[k-1:k+2]+1e-30)
    return (k+.5*(a-c)/(a-2*b+c))*SR/n
