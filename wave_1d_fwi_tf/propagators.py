"""1D finite difference wave propagation implemented using TensorFlow
"""
import numpy as np
import tensorflow as tf

class Propagator(object):
    """An 8th order finite difference propagator for the 1D wave equation."""
    def __init__(self, model, dx, dt=None, npad=8):
        self.nx = len(model)
        self.dx = np.float32(dx)
        max_vel = np.max(model)
        if dt:
            self.dt = dt
        else:
            self.dt = 0.6 * self.dx / max_vel
        self.nx_padded = self.nx + 2*npad
        self.model_padded = np.pad(model, (npad, npad), 'edge')
        self.model_padded2_dt2 = self.model_padded**2 * self.dt**2
        self.wavefield = [np.zeros(self.nx_padded, np.float32),
                          np.zeros(self.nx_padded, np.float32)
                         ]
        self.current_wavefield = self.wavefield[0]
        self.previous_wavefield = self.wavefield[1]

class VTF(Propagator):
    """A TensorFlow implementation."""
    def __init__(self, model, dx, dt=None):
        super(VTF, self).__init__(model, dx, dt, npad=0)
        # Create a TensorFlow Session
        self.sess = tf.Session()

        # Save c(x)^2 * dt^2 as a constant Tensor
        self.model_padded2_dt2 = tf.constant(self.model_padded2_dt2)

        # Create placeholders that will hold:
        #   Two time steps of the wavefield
        self.f = tf.placeholder(tf.float32, shape=(self.nx_padded))
        self.fp = tf.placeholder(tf.float32, shape=(self.nx_padded))
        #   The source amplitude with time, and the source positions
        self.sources = tf.placeholder(tf.float32, shape=(None))
        self.sources_x = tf.placeholder(tf.int64, shape=(None))

        # Create the spatial finite difference kernel that will calculate
        # d^2/dx^2, reshape it into the appropriate shape for a 1D
        # convolution, and save it as a constant tensor
        fd_kernel = np.array([-735, +15360,
                              -156800, +1053696,
                              -5350800, +22830080,
                              -94174080, +538137600,
                              -924708642,
                              +538137600, -94174080,
                              +22830080, -5350800,
                              +1053696, -156800,
                              +15360, -735] / (302702400 * self.dx**2),
                              np.float32)
        fd_kernel = fd_kernel.reshape([-1, 1, 1])
        fd_kernel = tf.constant(fd_kernel)

        # Calculate d^2/dx^2 by convolving with the above kernel
        def laplace(x):
            return tf.squeeze(tf.nn.conv1d(tf.reshape(x, [1, -1, 1]), fd_kernel, 1, 'SAME'))

        # The main evolution equation:
        # f(t+1, x) = c(x)^2 * dt^2 * d^2(f(t, x))/dx^2 + 2f(t, x) - f(t-1, x)
        self.fp_ = self.model_padded2_dt2 * laplace(self.f) + 2*self.f - self.fp

        # Add the sources
        # f(t+1, x_s) += c(x_s)^2 * dt^2 * s(t)
        # To do this, we need to extract c(x)^2 * dt^2 at the locations of the
        # sources. I do this using tf.gather, and then create a new array that
        # contains the source amplitudes multiplied by the appropriate
        # c(x)^2 * dt^2
        sources_v = tf.gather(self.model_padded2_dt2, self.sources_x)
        sources_amp = self.sources * sources_v
        # We can then add this new array to f(t+1), but to do so we need to
        # expand it into the same size as f(t+1) (it currently contains
        # one element for each source, but we need it to contain one element
        # for each x), so we use tf.sparse_to_dense. This will create an array
        # of the right size, almost entirely filled with zeros, with the
        # source amplitudes in the right places.
        self.fp_ += tf.sparse_to_dense(self.sources_x, [self.nx_padded], sources_amp)

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield one time step."""

        # tf.sparse_to_dense requires that the indicies of the source positions
        # (sources_x) are in order, so we need to sort them (and thus also the
        # source amplitudes (sources))
        ssort = sources_x.argsort()
        sources_sort = sources[ssort, :]
        sources_x_sort = sources_x[ssort]

        for istep in range(num_steps):
            # Extract the source amplitudes for this step
            sources_step = sources_sort[:,istep]

            # Run the computational graph to get the wavefield at t+1.
            # We need to pass in values for the placeholders.
            y = self.sess.run(self.fp_, {self.sources: sources_step,
                                         self.sources_x: sources_x_sort,
                                         self.f: self.current_wavefield,
                                         self.fp: self.previous_wavefield})
            # Save the calculated wavefield
            self.previous_wavefield[:] = y[:]

            # Swap the wavefield pointers in preparation for the next time step
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        if num_steps > 0:
            return y
        else:
            return self.current_wavefield
