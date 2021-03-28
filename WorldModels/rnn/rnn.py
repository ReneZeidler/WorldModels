from argparse import Namespace
from enum import IntFlag

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from losses import masked_gaussian_mixture, masked_bce, masked_mse, masked_bool_statistic, masked_bool_logvar


class FeatureMode(IntFlag):
    """
    Determines which features are used to train the controller.
    z: Encoding (from the Autoencoder)
    h: Hidden state (LSTM output)
    c: Cell state (LSTM state)
    """
    INCLUDE_Z = 2 ** 0
    INCLUDE_C = 2 ** 1
    INCLUDE_H = 2 ** 2
    MODE_ZCH = INCLUDE_Z | INCLUDE_C | INCLUDE_H
    MODE_ZH = INCLUDE_Z | INCLUDE_H
    MODE_ZC = INCLUDE_Z | INCLUDE_C
    MODE_Z = INCLUDE_Z

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return cls.from_str(value)
        return super()._missing_(value)

    @staticmethod
    def from_str(value: str):
        value = value.upper()
        return FeatureMode[value]


@tf.function
def sample_vae(vae_mu, vae_logvar):
    sz = vae_mu.shape[1]
    mu_logvar = tf.concat([vae_mu, vae_logvar], axis=1)
    z = tfp.layers.DistributionLambda(
        lambda theta: tfp.distributions.MultivariateNormalDiag(loc=theta[:, :sz], scale_diag=tf.exp(theta[:, sz:])),
        dtype=tf.float16
    )  # TODO: Don't instantiate new layer each time
    return z(mu_logvar)


class MDNRNN(tf.keras.Model):
    def __init__(self, args: Namespace):
        super(MDNRNN, self).__init__()
        self.args = args

        if self.args.rnn_decay_rate == 1.0:
            lr = self.args.rnn_learning_rate
        else:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                self.args.rnn_learning_rate, self.args.rnn_epoch_steps, self.args.rnn_decay_rate
            )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=self.args.rnn_grad_clip)

        self.loss_fn = self.get_loss()
        self.num_units = args.rnn_size

        # layers.LSTM can take advantage of CuDNN for a big performance boost
        self.inference_base = tf.keras.layers.LSTM(self.num_units, return_sequences=True, return_state=True)

        self.predict_z = tf.keras.Sequential([
            tf.keras.layers.Dense(args.rnn_out_size, input_shape=(self.num_units,), name="dense_z")
        ], name="predict_z")
        self.predict_done = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(self.num_units,), name="dense_done")
        ], name="predict_done") if self.args.rnn_predict_done else None
        self.predict_reward = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(self.num_units,), name="dense_reward")
        ], name="predict_reward") if self.args.rnn_predict_reward else None

        # TODO: Is this call necessary?
        super().build((self.args.rnn_batch_size, self.args.rnn_max_seq_len, self.args.rnn_input_seq_width))

    """
    Construct a loss functions for the MDN layer parametrised by number of mixtures.
    """
    def get_loss(self):
        # TODO: Named losses with dict
        num_mixtures = self.args.rnn_num_mixture
        batch_size = self.args.rnn_batch_size
        z_size = self.args.z_size
        losses = [masked_gaussian_mixture(num_mixtures, batch_size, z_size)]
        if self.args.rnn_predict_done:
            one_weight = 0.99
            zero_weight = 0.01
            losses.append(masked_bce(one_weight, zero_weight))
        if self.args.rnn_predict_reward:
            losses.append(masked_mse)
        losses = tuple(losses) if len(losses) > 1 else losses[0]
        return losses

    def get_metrics(self):
        if self.args.rnn_predict_done:
            return {"output_2": [masked_bool_statistic(True, "d_one_values"),
                                 masked_bool_statistic(False, "d_zero_values"),
                                 masked_bool_statistic(True, "d_one_accuracy", calc_accuracy=True),
                                 masked_bool_statistic(False, "d_zero_accuracy", calc_accuracy=True),
                                 masked_bool_logvar("d_logvar")]}  # TODO: Named losses
        else:
            return None

    # Never actually called
    def set_random_params(self, stddev=0.5):
        params = self.get_weights()
        rand_params = []
        for param_i in params:
            # David's spicy initialization scheme is wild but from preliminary experiments is critical
            sampled_param = np.random.standard_cauchy(param_i.shape) * stddev / 10000.0
            rand_params.append(sampled_param)  # spice things up

        self.set_weights(rand_params)

    def call(self, inputs, training=True, mask=None):
        assert mask is None, 'Unsupported argument "mask"'

        # whole_seq_output,        final_memory_state, final_carry_state
        # (batch, seq_len, units), (batch, units),     (batch, units)
        rnn_out, _state_h, _state_c = self.inference_base(inputs)
        # (batch * seq_len, units)
        rnn_out = tf.reshape(rnn_out, [-1, self.num_units])

        # TODO: named outputs with dict
        outputs = [self.predict_z(rnn_out)]
        if self.args.rnn_predict_done:
            outputs.append(self.predict_done(rnn_out))
        if self.args.rnn_predict_reward:
            outputs.append(self.predict_reward(rnn_out))

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)


@tf.function
def rnn_next_state(rnn, z, a, prev_state):
    z = tf.cast(tf.reshape(z, [1, 1, -1]), tf.float32)
    a = tf.cast(tf.reshape(a, [1, 1, -1]), tf.float32)
    z_a = tf.concat([z, a], axis=2)
    _rnn_out, h, c = rnn.inference_base(z_a, initial_state=prev_state)
    return [h, c]


@tf.function
def rnn_init_state(rnn):
    return rnn.inference_base.cell.get_initial_state(batch_size=1, dtype=tf.float32)


def rnn_output(rnn_state, z, mode: FeatureMode):
    h, c = rnn_state[0], rnn_state[1]

    # rnn_state must be from a batch with a single time series (for which we want to extract features)
    assert h.shape[0] == 1, "Tried to get features from batch > 1"
    assert c.shape[0] == 1, "Tried to get features from batch > 1"
    # Extract first (and only) time series in batch
    h = h[0]
    c = c[0]
    # We also expect z to be a batch with one element
    assert z.shape[0] == 1, "Expected z to be of shape (1, z_size), got " + str(z.shape)
    z = z[0]

    # Order taken from original implementation
    outputs = []
    if mode & FeatureMode.INCLUDE_Z:
        outputs.append(z)
    if mode & FeatureMode.INCLUDE_C:
        outputs.append(c)
    if mode & FeatureMode.INCLUDE_H:
        outputs.append(h)

    return tf.concat(outputs, axis=0).numpy()


def rnn_output_size(rnn_state_size: int, z_size: int, mode: FeatureMode) -> int:
    num_features = 0
    if mode & FeatureMode.INCLUDE_Z:
        num_features += z_size
    if mode & FeatureMode.INCLUDE_C:
        num_features += rnn_state_size
    if mode & FeatureMode.INCLUDE_H:
        num_features += rnn_state_size

    return num_features


@tf.function
def rnn_sim(rnn: MDNRNN, z, states, a):
    # Make one LSTM step
    z = tf.reshape(tf.cast(z, dtype=tf.float32), (1, 1, rnn.args.z_size))
    a = tf.reshape(tf.cast(a, dtype=tf.float32), (1, 1, rnn.args.a_width))
    input_x = tf.concat((z, a), axis=2)
    rnn_out, h, c = rnn.inference_base(input_x, initial_state=states)
    rnn_state = [h, c]
    rnn_out = tf.reshape(rnn_out, [-1, rnn.args.rnn_size])

    # Predict z
    mdnrnn_params = rnn.predict_z(rnn_out)
    mdnrnn_params = tf.reshape(mdnrnn_params, [-1, 3 * rnn.args.rnn_num_mixture])
    mu, logstd, logpi = tf.split(mdnrnn_params, num_or_size_splits=3, axis=1)
    logpi = logpi - tf.reduce_logsumexp(input_tensor=logpi, axis=1, keepdims=True)  # normalize

    cat = tfd.Categorical(logits=logpi)
    component_splits = [1] * rnn.args.rnn_num_mixture
    mus = tf.split(mu, num_or_size_splits=component_splits, axis=1)
    sigs = tf.split(tf.exp(logstd), num_or_size_splits=component_splits, axis=1)
    coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale in zip(mus, sigs)]
    mixture = tfd.Mixture(cat=cat, components=coll)

    z = tf.reshape(mixture.sample(), shape=(-1, rnn.args.z_size))

    # Predict done
    if rnn.args.rnn_predict_done:
        d_distr = rnn.predict_done(rnn_out)
        done_logit = tfd.Normal(d_distr[0][0], d_distr[0][1]).sample()
        done_dist = tfd.Binomial(total_count=1, logits=done_logit)
        done = tf.squeeze(done_dist.sample()) == 1.0
    else:
        done = False

    # Predict reward
    if rnn.args.rnn_predict_reward:
        r_distr = rnn.predict_reward(rnn_out)
        reward = tfd.Normal(r_distr[0][0], r_distr[0][1]).sample()
    else:
        reward = 1.0

    return rnn_state, z, reward, done
