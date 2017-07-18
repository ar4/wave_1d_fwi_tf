"""Test the inversion."""
import pytest
import numpy as np
from wave_1d_fd_pml.propagators import Pml
from wave_1d_fwi_tf.fwi import TFFWI

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = np.arange(-peak_time, (length)*dt - peak_time, dt, dtype=np.float32)
    y = ((1.0 - 2.0*(np.pi**2)*(freq**2)*(t**2))
         * np.exp(-(np.pi**2)*(freq**2)*(t**2)))
    return y

@pytest.fixture
def model_one(nsteps=None):
    """Create a model with one reflector."""
    N = 100
    rx = int(N/2)
    model = np.ones(N, dtype=np.float32) * 1500
    model[rx:] = 2500
    max_vel = 2500
    dx = 5
    dt = 0.001
    if nsteps==None:
        nsteps = np.ceil(2.5*rx*dx/1500/dt).astype(np.int)
    source = np.array([ricker(25, nsteps, dt, 0.05)]).reshape([1,-1])
    sx = np.array([1])
    rx = np.array([1, N-1])
    v = Pml(model, dx, dt=dt, pml_width=6, profile=1000/6*np.arange(6, dtype=np.float32))
    data = np.zeros([2, nsteps], dtype=np.float32)
    for i in range(nsteps):
        y = v.step(1, source[:,i:i+1], sx)
        data[0, i] = y[rx[0]]
        data[1, i] = y[rx[1]]
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': source, 'sx': sx, 'rx': rx,
            'data': data}

def test_one_reflector(model_one):
    """Verify that the inverted and true models are similar."""
    y = TFFWI(np.ones(len(model['model']))*1500,
              model['sources'], model['sx'],
              model['data'], model['rx'], model['nsteps'],
              model['dx'], model['dt'])
    assert np.allclose(y, model['model'], atol=1)
