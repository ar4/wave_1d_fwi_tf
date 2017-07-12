"""Test the inversion."""
import pytest
import numpy as np
from wave_1d_fwi_tf.propagators import VTF

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = np.arange(-peak_time, (length)*dt - peak_time, dt, dtype=np.float32)
    y = ((1.0 - 2.0*(np.pi**2)*(freq**2)*(t**2))
         * np.exp(-(np.pi**2)*(freq**2)*(t**2)))
    return y

@pytest.fixture
def model_one():
    """Create a model with one reflector."""
    N = 100
    rx = int(N/2)
    model = np.ones(N, dtype=np.float32) * 1500
    model[rx:] = 2500
    max_vel = 2500
    dx = 5
    dt = 0.001
    nsteps = np.ceil(2.5*rx*dx/1500/dt).astype(np.int)
    source = ricker(25, nsteps, dt, 0.05)
    sx = 1
    v = VTF(model, dx, dt)
    data = np.zeros(nsteps, dtype=np.float32)
    for i in range(nsteps):
        y = v.step(1, source[i], sx)
        data[i] = y[sx]
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx]),
            'data': data}

def test_one_reflector(model_one, versions):
    """Verify that the numeric and analytic wavefields are similar."""
    y = TFFWI(np.ones(len(model['model']))*1500, model['dx'], model['dt'],
              model['data'])
    assert np.allclose(y, model['model'], atol=1)
