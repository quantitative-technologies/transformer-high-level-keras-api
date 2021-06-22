import tensorflow as tf
from tensorflow.keras.layers import Layer

_keras_mask_attr = '_keras_mask'


class PaddingMask(Layer):
    def __init__(self, mask_value=0):
        super(PaddingMask, self).__init__()
        self.mask_value = mask_value

    def call(self, inputs):
        # Needed to ensure that mask gets recomputed
        if hasattr(inputs, _keras_mask_attr):
            delattr(inputs, _keras_mask_attr)
        return inputs

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, self.mask_value)


def create_look_ahead_mask(size):
    """
    Lower-triangular boolean matrix
    """

    mask = tf.cast(tf.linalg.band_part(tf.ones((size, size)), -1, 0), tf.bool)
    return mask  # (size, size)


class AutoRegressiveMask(Layer):
    def __init__(self):
        super(AutoRegressiveMask, self).__init__()

    def call(self, inputs, mask=None):
        # Needed to ensure that mask gets recomputed
        if hasattr(inputs, _keras_mask_attr):
            delattr(inputs, _keras_mask_attr)
        return inputs

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        seq_length = tf.shape(inputs)[1] #inputs.shape[1]

        look_ahead_mask = create_look_ahead_mask(seq_length)
        mask = tf.logical_and(look_ahead_mask, tf.expand_dims(mask, axis=1))
        return mask
