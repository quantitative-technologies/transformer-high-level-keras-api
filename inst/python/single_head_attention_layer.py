import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

from transformer.attention import scaled_dot_product_attention, MultiHeadAttention


class SingleHeadAttention(Layer):
    def __init__(self, d_model):
        super(SingleHeadAttention, self).__init__()
        self.d_model = d_model
        self.dense = Dense(d_model)

    def call(self, inputs, mask=None):
        q, k, v = inputs['q'], inputs['k'], inputs['v']

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class SingleHeadSelfAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads, mask_rank, kernel_initializer='glorot_uniform'):
        super(SingleHeadSelfAttention, self).__init__(d_model, num_heads, kernel_initializer)
        self._mask_rank = mask_rank

    def call(self, inputs, mask=None):
        if self._mask_rank == 2:
            mask = tf.expand_dims(mask, axis=1)
        output, attention_weights = super(SingleHeadSelfAttention, self) \
            .call(inputs=dict(q=inputs, k=inputs, v=inputs), mask=mask)

        return output


if __name__ == "__main__":
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    inputs = dict(q=y, k=y, v=y)
    out, attn = temp_mha(inputs, mask=None)
    out.shape, attn.shape

    temp_sha = SingleHeadAttention(d_model=512)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    inputs = dict(q=y, k=y, v=y)
    out, attn = temp_sha(inputs, mask=None)
    out.shape, attn.shape
