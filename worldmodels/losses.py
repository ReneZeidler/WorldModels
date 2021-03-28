import numpy as np
import tensorflow as tf


def reshape_and_split_data_mask(data, size_without_mask: int = 1):
    data = tf.reshape(data, [-1, size_without_mask + 1])
    return data[:, :-1], data[:, -1:]


def masked_average(data, mask):
    data = mask * data
    return tf.reduce_sum(data) / tf.reduce_sum(mask)  # mean of unmasked


def sample_mu_logvar(pred):
    mu, logvar = tf.split(pred, 2, axis=-1)
    eps = tf.random.normal(shape=tf.shape(mu))
    return eps * tf.exp(logvar * .5) + mu


def masked_gaussian_mixture(num_mixtures: int, batch_size: int, z_size: int):
    """
    This loss function is defined for N*k components each containing a gaussian of 1 feature
    """

    def loss_func(y_true, y_pred):
        # Flatten timestamp axis and separate mask from target output
        z_true, mask = reshape_and_split_data_mask(y_true, z_size)

        z_true = tf.reshape(z_true, [-1, 1])  # Flatten z-element axis
        mask = tf.reshape(mask, [batch_size, -1, 1])
        mask = tf.tile(mask, [1, 1, z_size])  # Repeat mask value for every z element
        mask = tf.reshape(mask, [-1, 1])  # Flatten timestep and tiled dimension into batch dimension

        # Reshape inputs in case this is used in a TimeDistributed layer
        y_pred = tf.reshape(y_pred, [-1, 3 * num_mixtures], name='reshape_ypreds')

        # Interpret outputs as a Gaussian mixture and calculate loss
        out_mu, out_logstd, out_logp = tf.split(y_pred, num_or_size_splits=3, axis=1, name='mdn_coef_split')
        out_logp = out_logp - tf.reduce_logsumexp(input_tensor=out_logp, axis=1, keepdims=True)  # normalize

        log_sqrt_2pi = np.log(np.sqrt(2.0 * np.pi))
        lognormal = -0.5 * ((z_true - out_mu) / tf.exp(out_logstd)) ** 2 - out_logstd - log_sqrt_2pi
        v = out_logp + lognormal
        z_loss = -tf.reduce_logsumexp(input_tensor=v, axis=1, keepdims=True)

        # Apply mask to not propagate loss for masked inputs
        return masked_average(z_loss, mask)

    return loss_func


def masked_bce(one_weight: float = 1, zero_weight: float = 1):
    def loss_func(y_true, y_pred):
        # Flatten timestamp axis and separate mask from target output
        d_true, mask = reshape_and_split_data_mask(y_true)

        # Sample predicted done flag logit from normal distribution
        d_pred = sample_mu_logvar(y_pred)

        # Cross entropy between actual done flag and predicted done flag (log probability)
        d_loss = tf.keras.losses.binary_crossentropy(y_true=d_true, y_pred=d_pred, from_logits=True)
        d_loss = tf.expand_dims(d_loss, axis=-1)

        # Add class weights to incentivize learning done=True more
        weight_vector = d_true * one_weight + (1. - d_true) * zero_weight
        weight_vector *= tf.cast(tf.size(weight_vector), tf.float32) / tf.reduce_sum(weight_vector)
        d_loss = weight_vector * d_loss

        # Apply mask to not propagate loss for masked inputs
        return masked_average(d_loss, mask)

    return loss_func


def masked_mse(y_true, y_pred):
    # Flatten timestamp axis and separate mask from target output
    r_true, mask = reshape_and_split_data_mask(y_true)

    # Sample predicted reward from normal distribution
    r_pred = sample_mu_logvar(y_pred)

    # Mean squared error loss
    r_loss = tf.keras.losses.mean_squared_error(y_true=r_true, y_pred=r_pred)
    r_loss = tf.expand_dims(r_loss, axis=-1)

    # Apply mask to not propagate loss for masked inputs
    return masked_average(r_loss, mask)


def masked_bool_statistic(for_target: bool, name: str, calc_accuracy: bool = False, thresh_stddev: float = 2):
    def metric(y_true, y_pred):
        # Flatten timestamp axis and separate mask from target output
        true_bool, mask = reshape_and_split_data_mask(y_true)
        mu, logvar = tf.split(y_pred, 2, axis=-1)

        # Unmasked real done flags
        if for_target is False:
            true_bool = 1.0 - true_bool
        mask = mask * true_bool

        if calc_accuracy:
            # Set to True (1) where mu is at least thresh_stddev sigmas above/below 0 (using logvar)
            thresh = tf.sqrt(tf.exp(logvar * 0.5)) * thresh_stddev
            mu = mu > thresh if for_target is True else mu < thresh
            mu = tf.cast(mu, tf.float32)

        # Average value of outputs where done is equal to for_target
        return masked_average(mu, mask)
    metric.__name__ = name
    return metric


def masked_bool_logvar(name: str):
    def metric(y_true, y_pred):
        # Flatten timestamp axis and separate mask from target output
        _true_bool, mask = reshape_and_split_data_mask(y_true)
        _mu, logvar = tf.split(y_pred, 2, axis=-1)
        return masked_average(logvar, mask)
    metric.__name__ = name
    return metric