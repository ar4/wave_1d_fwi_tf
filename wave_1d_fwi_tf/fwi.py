"""1D FWI implemented using TensorFlow
"""
import numpy as np
import tensorflow as tf

class Invertor(object):
    """A full waveform invertor for the 1D wave equation."""
    def __init__(self, model, dx, dt, npad=8):
        self.nx = len(model)
        self.dx = np.float32(dx)
        self.dt = dt
        self.nx_padded = self.nx + 2*npad
        self.model_padded = np.pad(model, (npad, npad), 'edge')
        self.wavefield = [np.zeros(self.nx_padded, np.float32),
                          np.zeros(self.nx_padded, np.float32)
                         ]
        self.current_wavefield = self.wavefield[0]
        self.previous_wavefield = self.wavefield[1]

class TFFWI(Invertor):
    """A TensorFlow implementation."""
    def __init__(self, model, sources, sources_x, receivers, receivers_x, nsteps, dx, dt=None, pml_width=6, profile=None):
        super(TFFWI, self).__init__(model, dx, dt, npad=pml_width)
        # Create a TensorFlow Session
        self.sess = tf.Session()

        # c(x) is the Variable we are inverting for
        self.model_padded = tf.Variable(self.model_padded)
        model_padded2_dt2 = tf.square(self.model_padded) * dt**2

        if profile is None:
            profile = 1000/pml_width*np.arange(pml_width, dtype=np.float32)

        sigma = np.zeros(self.nx_padded, np.float32)
        sigma[pml_width-1::-1] = profile
        sigma[-pml_width:] = profile

        sigma = tf.constant(sigma)

        # Create placeholders that will hold:
        #   Two time steps of the wavefield
        f = tf.zeros([self.nx_padded])
        fp = tf.zeros([self.nx_padded])
        #   Two time steps of phi
        phi = tf.zeros([self.nx_padded])
        phip = tf.zeros([self.nx_padded])
        #   The source amplitude with time, and the source positions
        # tf.sparse_to_dense requires that the indicies of the source positions
        # (sources_x) are in order, so we need to sort them (and thus also the
        # source amplitudes (sources))
        ssort = sources_x.argsort()
        sources_sort = sources[ssort, :]
        sources_x_sort = sources_x[ssort]

        sources = tf.constant(sources_sort)
        sources_x = tf.constant(sources_x_sort + pml_width)
        sources_v = tf.gather(model_padded2_dt2, sources_x)

        rsort = receivers_x.argsort()
        receivers_sort = receivers[rsort, :]
        receivers_x_sort = receivers_x[rsort]

        receivers = tf.constant(receivers_sort)
        receivers_x = tf.constant(receivers_x_sort + pml_width)

        # Create the spatial finite difference kernel that will calculate
        # d/dx, reshape it into the appropriate shape for a 1D
        # convolution, and save it as a constant tensor
        d1_kernel = np.array([5, -72, 495, -2200, 7425, -23760, 0,
                              23760, -7425, 2200, -495, 72, -5] / (27720 * self.dx),
                              np.float32)
        d1_kernel = d1_kernel.reshape([-1, 1, 1])
        d1_kernel = tf.constant(d1_kernel)

        # Create the spatial finite difference kernel that will calculate
        # d^2/dx^2, reshape it into the appropriate shape for a 1D
        # convolution, and save it as a constant tensor
        d2_kernel = np.array([-735, +15360,
                              -156800, +1053696,
                              -5350800, +22830080,
                              -94174080, +538137600,
                              -924708642,
                              +538137600, -94174080,
                              +22830080, -5350800,
                              +1053696, -156800,
                              +15360, -735] / (302702400 * self.dx**2),
                              np.float32)
        d2_kernel = d2_kernel.reshape([-1, 1, 1])
        d2_kernel = tf.constant(d2_kernel)

        # Matrix to set gradient of model in PML regions to first non-PML cell
        border_mute = np.eye(self.nx_padded, dtype=np.float32)
        border_mute[:8,:] = 0
        border_mute[:8,8] = 1
        border_mute[-7:,:] = 0
        border_mute[-7:,-8] = 1
        border_mute = tf.constant(border_mute)

        # Calculate d/dx by convolving with the appropriate kernel
        def first_deriv(x):
            return tf.squeeze(tf.nn.conv1d(tf.reshape(x, [1, -1, 1]), d1_kernel, 1, 'SAME'))

        # Calculate d^2/dx^2 by convolving with the appropriate kernel
        def second_deriv(x):
            return tf.squeeze(tf.nn.conv1d(tf.reshape(x, [1, -1, 1]), d2_kernel, 1, 'SAME'))

        for step in range(nsteps):

            # The main evolution equation:
            # f(t+1, x) = c(x)^2 * dt^2 / (1 + dt * sigma(x)/2) * (d^2(f(t, x))/dx^2 + d(phi(t, x)/dx)
            # + dt * sigma(x) * f(t-1, x) / (2 + dt * sigma(x))
            # + 1 / (1 + dt * sigma(x) / 2) * (2 * f(t, x) - f(t-1, x))
            fp_ = (model_padded2_dt2 / (1 + dt * sigma/2)
                   * (second_deriv(f) + first_deriv(phi))
                   + dt * sigma * fp / (2 + dt * sigma)
                   + 1 / (1 + dt * sigma / 2) * (2 * f - fp))

            # phip(i) =  -sigma(i) * dt * f_x + phi(i) - dt * sigma(i) * phi(i)
            phip_ = -sigma * dt * first_deriv(f) + phi - dt * sigma * phi

            # Add the sources
            # f(t+1, x_s) += c(x_s)^2 * dt^2 * s(t)
            # To do this, we need to extract c(x)^2 * dt^2 at the locations of the
            # sources. I do this using tf.gather, and then create a new array that
            # contains the source amplitudes multiplied by the appropriate
            # c(x)^2 * dt^2
            sources_amp = sources[:, step] * sources_v
            # We can then add this new array to f(t+1), but to do so we need to
            # expand it into the same size as f(t+1) (it currently contains
            # one element for each source, but we need it to contain one element
            # for each x), so we use tf.sparse_to_dense. This will create an array
            # of the right size, almost entirely filled with zeros, with the
            # source amplitudes in the right places.
            fp_ += tf.sparse_to_dense(sources_x, [self.nx_padded], sources_amp)

            receivers_amp = tf.reshape(tf.gather(fp_, receivers_x), [-1, 1])
            if step==0:
                self.y = receivers_amp
            else:
                self.y = tf.concat([self.y, receivers_amp], axis=1)

            fp = f
            f = fp_
            phip = phi
            phi = phip_
            
        self.loss = tf.losses.mean_squared_error(receivers, self.y)
        trainer = tf.train.GradientDescentOptimizer(10000)
        #trainer = tf.train.AdamOptimizer(learning_rate=1)
        [(self.y_grad, self.y_val)]  = trainer.compute_gradients(self.loss)
        self.y_grad = tf.squeeze(tf.matmul(border_mute, tf.expand_dims(self.y_grad, 1)))
        self.optimizer = trainer.apply_gradients([(self.y_grad, self.y_val)])
        #self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def invert(self, nsteps=100, print_freq=100):
        self.sess.run(self.init)
        for step in range(nsteps):
            _, l, pred_y, pred_model = self.sess.run([self.optimizer, self.loss, self.y, self.model_padded])
            if step % np.maximum(int(np.round(nsteps/print_freq)),1) == 0:
              print(step, l)

        return pred_y, pred_model
            

