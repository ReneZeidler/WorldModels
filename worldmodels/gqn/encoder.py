import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D


class TowerEncoder(tf.keras.Model):
    def __init__(self, r_dim, pool: bool = False, name=None):
        super().__init__(name=name)
        self.pool = pool

        k = r_dim

        activation_f = tf.nn.relu

        self.c1 = Conv2D(k     , 2, strides=2,      activation=activation_f, name="enc1")
        self.c2 = Conv2D(k     , 2, strides=2,      activation=activation_f, name="enc23_skip")
        self.c3 = Conv2D(k // 2, 3, padding='SAME', activation=activation_f, name="enc2")
        self.c4 = Conv2D(k     , 2, strides=2,      activation=activation_f, name="enc3")

        self.c5 = Conv2D(k     , 3, padding='SAME', activation=activation_f, name="enc45_skip")
        self.c6 = Conv2D(k // 2, 3, padding='SAME', activation=activation_f, name="enc4")
        self.c7 = Conv2D(k     , 3, padding='SAME', activation=activation_f, name="enc5")
        self.c8 = Conv2D(k     , 1, padding='SAME', activation=activation_f, name="enc6")

        self.avg_pool = AveragePooling2D(16, name="avg_pool")

    def call(self, inputs, training=None, **kwargs):
        x, v = inputs

        batch_size, v_dim = v.shape
        batch_size = tf.shape(v)[0]  # Needs to be dynamic, batch size isn't known at model compile time
        # batch_size, im_size, im_size, x_dim = x.shape
        broadcast_size = 16

        v = tf.reshape(v, [batch_size, 1, 1, v_dim])
        v = tf.broadcast_to(v, [batch_size, broadcast_size, broadcast_size, v_dim])

        skip_in = self.c1(x)
        skip_out = self.c2(skip_in)

        x = self.c3(skip_in)
        x = self.c4(x) + skip_out

        skip_in = tf.concat([x, v], 3)
        skip_out = self.c5(skip_in)

        x = self.c6(skip_in)
        x = self.c7(x) + skip_out

        x = self.c8(x)

        if self.pool:
            # 16x16xr_dim -> 1x1xr_dim
            x = self.avg_pool(x)

        return x


class PyramidEncoder(tf.keras.Model):
    def __init__(self, r_dim, name=None):
        super().__init__(name=name)
        k = r_dim

        activation_f = tf.nn.relu

        # In: 64x64x(7+3)
        self.c1 = Conv2D(k // 8, 2, strides=2, activation=activation_f, name="enc1")  # 32x32x 32
        self.c2 = Conv2D(k // 4, 2, strides=2, activation=activation_f, name="enc2")  # 16x16x 64
        self.c3 = Conv2D(k // 2, 2, strides=2, activation=activation_f, name="enc3")  #  8x 8x128
        self.c4 = Conv2D(k // 1, 8, strides=8, activation=activation_f, name="enc4")  #  1x 1x256

    def call(self, inputs, training=None, **kwargs):
        x, v = inputs

        batch_size, v_dim = v.shape
        batch_size = tf.shape(v)[0]  # Needs to be dynamic, batch size isn't known at model compile time
        # batch_size, im_size, im_size, x_dim = x.shape
        broadcast_size = 64

        # Broadcast camera data to every pixel for the input image (64x64)
        v = tf.reshape(v, [batch_size, 1, 1, v_dim])
        v = tf.broadcast_to(v, [batch_size, broadcast_size, broadcast_size, v_dim])

        # Concat camera and image data
        x = tf.concat([v, x], 3)

        # Apply convolutional layers
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)

        return x
