import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

im_size   = 64  # Size of original images
work_size = 16  # Size of images in generator cells (originally 16)
scale = im_size // work_size


class Conv2dLSTMCell(tf.keras.Model):
    def __init__(self, latent_dim, kernel_size=5, name=None):  # kernel_size=5
        super().__init__(name=name)

        args = [latent_dim, kernel_size]
        kwargs = {'padding': 'SAME'}

        self.forget = Conv2D(*args, **kwargs, activation=tf.sigmoid, name="forget")
        self.inp    = Conv2D(*args, **kwargs, activation=tf.sigmoid, name="inp"   )
        self.outp   = Conv2D(*args, **kwargs, activation=tf.sigmoid, name="outp"  )
        self.state  = Conv2D(*args, **kwargs, activation=tf.tanh   , name="state" )

    def call(self, inputs, **kwargs):
        input, cell = inputs

        forget_gate = self.forget(input)
        input_gate  = self.inp   (input)
        output_gate = self.outp  (input)
        state_gate  = self.state (input)

        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * tf.tanh(cell)

        return hidden, cell


class LatentDistribution(tf.keras.Model):
    def __init__(self, z_dim, name=None):
        super().__init__(name=name)
        self.z_dim = z_dim
        self.parametrize = Conv2D(z_dim * 2, 5, padding='SAME')

    def call(self, input, **kwargs):
        parametrization = self.parametrize(input)
        mu, sigma = tf.split(parametrization, [self.z_dim, self.z_dim], -1)
        return tfp.distributions.Normal(loc=mu, scale=tf.nn.softplus(sigma))


class Generator(tf.keras.Model):
    def __init__(self, x_dim, z_dim, h_dim, L, name=None):
        super().__init__(name=name)
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.inference_core = Conv2dLSTMCell(h_dim, name="inference_core")
        self.generator_core = Conv2dLSTMCell(h_dim, name="generator_core")

        self.posterior_distribution = LatentDistribution(z_dim, name="posterior")
        self.prior_distribution     = LatentDistribution(z_dim, name="prior"    )

        self.observation_density = Conv2D(x_dim, 1, padding='SAME', activation=tf.sigmoid, name="obs_density")

        self.upsample   = Conv2DTranspose(h_dim, scale, strides=scale, name="upsample"  )
        self.downsample = Conv2D         (h_dim, scale, strides=scale, name="downsample")

    def call(self, inputs, training=None, **kwargs):
        # target image, query camera, representation
        x, v, r = inputs

        batch_size, v_dim                 = v.shape
        batch_size, r_size, r_size, r_dim = r.shape
        # batch_size, im_size, im_size, x_dim = x.shape
        batch_size = tf.shape(v)[0]  # Needs to be dynamic, batch size isn't known at model compile time

        # Broadcast camera data to every pixel for the working data
        v = tf.tile   (v, [ 1, work_size * work_size      ])
        v = tf.reshape(v, [-1, work_size, work_size, v_dim])
        # Broadcast representation to every pixel for the working data
        # (assumes r_size == 1, i.e. Pyramid or Pool model for the encoder)
        r = tf.broadcast_to(r, [batch_size, work_size, work_size, r_dim])

        # Downsample image to working size
        x = self.downsample(x)

        # Hidden and cell state for LSTM
        c_g = tf.zeros([batch_size, work_size, work_size, self.h_dim])
        h_g = tf.zeros([batch_size, work_size, work_size, self.h_dim])

        c_i = tf.zeros([batch_size, work_size, work_size, self.h_dim])
        h_i = tf.zeros([batch_size, work_size, work_size, self.h_dim])

        # Work image
        u   = tf.zeros([batch_size, im_size,   im_size,   self.h_dim])

        kl = 0
        for _ in range(self.L):
            # Prior factor (eta π network)
            prior_factor = self.prior_distribution(h_g)

            # Inference state update
            input = tf.concat([h_i, h_g, x, v, r], 3)
            h_i, c_i = self.inference_core((input, c_i))

            # Posterior factor (eta e network)
            posterior_factor = self.posterior_distribution(h_i)

            # Posterior sample
            z = posterior_factor.sample()

            # Generator state update
            input = tf.concat([h_g, z, v, r], 3)
            h_g, c_g = self.generator_core((input, c_g))

            # Upsample and add to working image
            u = self.upsample(h_g) + u

            # Calculate KL-divergence
            kl += tf.reduce_mean(tfp.distributions.kl_divergence(posterior_factor, prior_factor))

        x_mu = self.observation_density(u)

        return x_mu, kl

    def sample(self, inputs):
        # query camera, representation
        v, r = inputs

        batch_size, v_dim                 = v.shape
        batch_size, r_size, r_size, r_dim = r.shape
        # batch_size, im_size, im_size, x_dim = x.shape
        batch_size = tf.shape(v)[0]  # Needs to be dynamic, batch size isn't known at model compile time

        # Broadcast camera data to every pixel for the working data
        v = tf.tile   (v, [ 1, work_size * work_size      ])
        v = tf.reshape(v, [-1, work_size, work_size, v_dim])
        # Broadcast representation to every pixel for the working data (assumes r_size == 1, i.e. PyramidEncoder)
        r = tf.broadcast_to(r, [batch_size, work_size, work_size, r_dim])

        # Hidden and cell state for LSTM
        c_g = tf.zeros([batch_size, work_size, work_size, self.h_dim])
        h_g = tf.zeros([batch_size, work_size, work_size, self.h_dim])

        # Work image
        u   = tf.zeros([batch_size, im_size,   im_size,   self.h_dim])

        for _ in range(self.L):
            prior_factor = self.prior_distribution(h_g)
            # Prior sample
            z = prior_factor.sample()
            # Generate state update
            input = tf.concat([h_g, z, v, r], 3)
            h_g, c_g = self.generator_core((input, c_g))
            # Update working image
            u = self.upsample(h_g) + u

        x_mu = self.observation_density(u)

        return x_mu
