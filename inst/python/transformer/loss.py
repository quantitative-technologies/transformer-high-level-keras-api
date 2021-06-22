import tensorflow as tf
from tensorflow.keras.losses import Loss, sparse_categorical_crossentropy


class MaskedSparseCategoricalCrossentropy(Loss):
    def __init__(self, name='masked_sparse_categorical_cross_entropy'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss = sparse_categorical_crossentropy(y_true, y_pred,
                                               from_logits=True)
        mask = getattr(y_pred, '_keras_mask')
        sw = tf.cast(mask, y_pred.dtype)
        # desired loss value
        reduced_loss = tf.reduce_sum(loss * sw) / tf.reduce_sum(sw)
        # cannot opt out of mask corrections in the API
        correction_factor = tf.reduce_sum(tf.ones(shape=tf.shape(y_true))) / \
                            tf.reduce_sum(sw)

        return reduced_loss * correction_factor

