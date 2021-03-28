import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

from gqn.encoder import PyramidEncoder, TowerEncoder
from gqn.generator import Generator


class GenerativeQueryNetwork(tf.keras.Model):
    # noinspection PyPep8Naming
    def __init__(self, x_dim, r_dim, h_dim, z_dim, L=12, name=None):
        super().__init__(name=name)

        self.r_dim = r_dim
        # self.encoder  = PyramidEncoder(r_dim, name="pyramid_encoder")
        # self.encoder  = TowerEncoder(r_dim, name="tower_encoder")
        self.encoder  = TowerEncoder(r_dim, pool=True, name="pool_encoder")
        self.generate = Generator(x_dim, z_dim, h_dim, L, name="generator")

        self.optimizer = None
        self.sigma = None
        self.const_sigma = 1.0

        # Loss is manually calculated in training loop, we need this to avoid errors
        self.loss_functions = []

    def call(self, batch, training=None, **kwargs):
        context_frames  = batch["context_frames" ]
        context_cameras = batch["context_cameras"]
        batch_size, context_size, *x_shape = context_frames .shape
        batch_size, context_size, *v_shape = context_cameras.shape

        x = tf.reshape(context_frames,  [-1, *x_shape], name="gather_frames" )
        v = tf.reshape(context_cameras, [-1, *v_shape], name="gather_cameras")

        phi = self.encoder((x, v))

        _, *phi_shape = phi.shape
        phi = tf.reshape(phi, [-1, context_size, *phi_shape], name="regroup_representation")

        r = tf.reduce_sum(phi, axis=1, name="sum_representation")

        x_q, v_q = batch["target"], batch["query_camera"]
        x_mu, kl = self.generate((x_q, v_q, r))

        return x_mu, x_q, r, kl

    def sample(self, batch):
        context_frames  = batch["context_frames" ]
        context_cameras = batch["context_cameras"]
        batch_size, context_size, *x_shape = context_frames .shape
        batch_size, context_size, *v_shape = context_cameras.shape

        x = tf.reshape(context_frames,  [-1, *x_shape], name="gather_frames" )
        v = tf.reshape(context_cameras, [-1, *v_shape], name="gather_cameras")

        phi = self.encoder((x, v))

        _, *phi_shape = phi.shape
        phi = tf.reshape(phi, [-1, context_size, *phi_shape], name="regroup_representation")

        r = tf.reduce_sum(phi, axis=1, name="sum_representation")

        x_q, v_q = batch["target"], batch["query_camera"]
        x_mu = self.generate.sample((v_q, r))

        return x_mu, x_q, r

    def encode(self, image, camera):
        # Encodes batch as single context frames (each image in batch is encoded separately)
        # Representations are flattened
        batch_size = tf.shape(image)[0]
        r = self.encoder((image, camera))
        return tf.reshape(r, (batch_size, -1))

    # noinspection PyMethodOverriding
    def compile(self, optimizer, sigma, const_sigma: float = 1.0):
        super().compile()
        self.optimizer = optimizer
        self.sigma = sigma
        self.const_sigma = const_sigma
        # noinspection PyAttributeOutsideInit
        self._is_compiled = True

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            x_mu, x_q, r, kl = self(x, training=True)

            decayed_sigma = self.sigma(self.optimizer.iterations)
            output_dist = tfp.distributions.Normal(loc=x_mu, scale=decayed_sigma)
            log_likelihood = output_dist.log_prob(x_q, name="ll")
            log_likelihood = tf.reduce_logsumexp(log_likelihood, name="reduce_ll")

            output_dist_const_var = tfp.distributions.Normal(loc=x_mu, scale=self.const_sigma)
            log_likelihood_const_var = output_dist_const_var.log_prob(x_q, name="llc")
            log_likelihood_const_var = tf.reduce_logsumexp(log_likelihood_const_var, name="reduce_llc")

            # ELBO (evidence lower bound): log_likelihood - kl (== -loss)
            loss = tf.subtract(kl, log_likelihood, name="loss")

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return {
            "loss": loss,
            "kl": kl,
            "ll": log_likelihood,
            "llc": log_likelihood_const_var,
            "sigma": decayed_sigma,
        }

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        x_mu, x_q, r = self.sample(x)

        decayed_sigma = self.sigma(self.optimizer.iterations)
        output_dist = tfp.distributions.Normal(loc=x_mu, scale=decayed_sigma)
        log_likelihood = tf.reduce_logsumexp(output_dist.log_prob(x_q))

        output_dist_const_var = tfp.distributions.Normal(loc=x_mu, scale=self.const_sigma)
        log_likelihood_const_var = tf.reduce_logsumexp(output_dist_const_var.log_prob(x_q))

        return {
            "ll": log_likelihood,
            "llc": log_likelihood_const_var,
        }
