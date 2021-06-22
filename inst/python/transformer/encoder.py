import tensorflow as tf
from tensorflow.keras import Sequential
from transformer.positional_encoding import positional_encoding
from transformer.attention import MultiHeadSelfAttention
from transformer.feed_forward import pointwise_feed_forward_network
from transformer.mask import PaddingMask
from transformer.sublayer import TransformerSubLayer
from transformer.embedding import ScaledEmbedding
from single_head_attention_layer import SingleHeadSelfAttention


def encoder_layer(d_model, num_heads, dff, dropout_rate=0.1,
                  kernel_initializer='glorot_uniform'):
    mha = MultiHeadSelfAttention(d_model, num_heads, mask_rank=2,
                                  kernel_initializer=kernel_initializer)
    ffn = pointwise_feed_forward_network(d_model, dff,
                                         kernel_initializer=kernel_initializer)

    return tf.keras.Sequential([
        TransformerSubLayer(mha, epsilon=1e-6, dropout_rate=dropout_rate),
        TransformerSubLayer(ffn, epsilon=1e-6, dropout_rate=dropout_rate)
    ])


def encoder(num_layers, d_model, num_heads, dff, input_vocab_size,
            maximum_position_encoding, dropout_rate=0.1,
            embedding_initializer='uniform', kernel_initializer='glorot_uniform'):
    layer_list = [
        PaddingMask(mask_value=0),
        ScaledEmbedding(input_vocab_size, d_model, dropout_rate,
                        maximum_position_encoding, positional=True,
                        embedding_initializer=embedding_initializer)] + \
        [encoder_layer(d_model, num_heads, dff, dropout_rate,
                       kernel_initializer=kernel_initializer)
         for i in range(num_layers)]

    return Sequential(layer_list)
