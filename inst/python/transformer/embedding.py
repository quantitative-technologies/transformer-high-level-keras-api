import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dropout
from .positional_encoding import positional_encoding


class ScaledEmbedding(Layer):
    def __init__(self, input_dim, output_dim, dropout_rate, max_seqlen,
                 positional=True, embedding_initializer='uniform'):
        super(ScaledEmbedding, self).__init__()
        self.embedding = Embedding(input_dim, output_dim,
                                   embeddings_initializer=embedding_initializer,
                                   mask_zero=True)
        self.positional = positional
        if positional:
            self._positions_enc = positional_encoding(max_seqlen, output_dim)
        self.dropout = Dropout(dropout_rate)
        self._c = tf.math.sqrt(tf.cast(output_dim, dtype=tf.float32))
        self.supports_masking = True

    def call(self, inputs, training=None):
        x_enc = self.embedding(inputs) * self._c
        if self.positional:
            seq_len = tf.shape(inputs)[1]
            x_enc += self._positions_enc[:, :seq_len, :]
        return self.dropout(x_enc, training)

