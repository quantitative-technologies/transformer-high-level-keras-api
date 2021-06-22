import sys

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input
from transformer.attention import MultiHeadAttention, MultiHeadSelfAttention, MultiHeadTwoInputAttention
from transformer.feed_forward import pointwise_feed_forward_network
from transformer.positional_encoding import positional_encoding
from transformer.mask import AutoRegressiveMask, PaddingMask#, \
    #create_padding_mask, compute_autoregressive_mask
from transformer.sublayer import TransformerSubLayer
from single_head_attention_layer import SingleHeadAttention
from transformer.embedding import ScaledEmbedding
from single_head_attention_layer import SingleHeadSelfAttention


def decoder_layer(d_model, num_heads, dff, dropout_rate=0.1,
                  kernel_initializer='glorot_uniform'):
    encoder_output = Input(shape=(None, d_model), name='encoder_output')
    decoder_input = Input(shape=(None, d_model), name='decoder_input')

    auto_regress = AutoRegressiveMask()
    mha_self = MultiHeadSelfAttention(d_model, num_heads, mask_rank=4, kernel_initializer=kernel_initializer)
    mha_auto_reg = Sequential([auto_regress, mha_self])
    mha_self_sublayer = TransformerSubLayer(mha_auto_reg, epsilon=1e-6, dropout_rate=dropout_rate)

    mha_2inp = MultiHeadTwoInputAttention(d_model, num_heads)
    mha_2inp_sublayer = TransformerSubLayer(mha_2inp, input_key='queries', epsilon=1e-6, dropout_rate=dropout_rate)

    ffn = pointwise_feed_forward_network(d_model, dff)
    ffn_sublayer = TransformerSubLayer(ffn, epsilon=1e-6, dropout_rate=dropout_rate)

    out1 = mha_self_sublayer(decoder_input)
    out2 = mha_2inp_sublayer(dict(queries=out1, lookups=encoder_output))
    outputs = ffn_sublayer(out2)

    return Model(inputs=[encoder_output, decoder_input], outputs=outputs)


def decoder(num_layers, d_model, num_heads, dff, target_vocab_size,
            maximum_position_encoding, dropout_rate=0.1,
            embedding_initializer='uniform', kernel_initializer='glorot_uniform'):

    encoder_output = Input(shape=(None, d_model), name='encoder_output')
    decoder_input = Input(shape=(None, ), name='decoder_input')

    embedding = Sequential([PaddingMask(),
                            ScaledEmbedding(target_vocab_size,
                                            d_model, dropout_rate,
                                            maximum_position_encoding,
                                            positional=True,
                                            embedding_initializer=embedding_initializer)])
    decoder_layers = [
        decoder_layer(d_model, num_heads, dff, dropout_rate=dropout_rate,
                      kernel_initializer=kernel_initializer)
        for _ in range(num_layers)]

    x = embedding(decoder_input)
    for i in range(num_layers):
        x = decoder_layers[i](dict(decoder_input=x, encoder_output=encoder_output))

    return Model(inputs=[encoder_output, decoder_input], outputs=x)

