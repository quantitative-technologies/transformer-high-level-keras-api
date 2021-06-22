import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add -infinity to the masked out positions in the scaled tensor.
    if mask is not None:
        masked_out = tf.cast(tf.math.logical_not(mask), tf.float32)
        scaled_attention_logits += masked_out * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, kernel_initializer='glorot_uniform'):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.kernel_initializer = kernel_initializer

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)
        self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)

        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)
        
        self.supports_masking = True

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        q, k, v = inputs['q'], inputs['k'], inputs['v']
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        if mask is not None:
            tf.assert_rank(mask, rank=3, message=f'rank {tf.rank(mask)} mask')
            mask = tf.expand_dims(mask, axis=1)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads, mask_rank, kernel_initializer='glorot_uniform'):
        super(MultiHeadSelfAttention, self).__init__(d_model, num_heads, kernel_initializer)
        self._mask_rank = mask_rank

    def call(self, inputs, mask=None):
        # if mask is None:
        #     # We are in the graph building stage
        #     tf.print("Building graph")
        if self._mask_rank == 2:
            mask = tf.expand_dims(mask, axis=1)
        output, attention_weights = super(MultiHeadSelfAttention, self) \
            .call(inputs=dict(q=inputs, k=inputs, v=inputs), mask=mask)

        return output


class MultiHeadTwoInputAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads, mask_rank=2, kernel_initializer='glorot_uniform'):
        super(MultiHeadTwoInputAttention, self).__init__(d_model, num_heads, kernel_initializer)
        self._mask_rank = mask_rank

    def call(self, inputs, mask=None):
        # queries, lookups = inputs
        queries, lookups = inputs['queries'], inputs['lookups']

        if mask is not None: # and 'lookups' in mask:
            mask = mask['lookups']
            if self._mask_rank == 2 and mask is not None:
                mask = tf.expand_dims(mask, axis=1)
        output, attention_weights = super(MultiHeadTwoInputAttention, self) \
            .call(inputs=dict(q=queries, k=lookups, v=lookups), mask=mask)

        return output
    
    def compute_mask(self, inputs, mask=None):
        """
        Pass the queries mask downstream
        """
        return None if mask is None else mask['queries']

