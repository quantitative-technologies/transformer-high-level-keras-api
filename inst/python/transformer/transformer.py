#import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

from .decoder import decoder
from .encoder import encoder


def transformer(num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size,
                 pe_input_max, pe_target_max, dropout_rate=0.1,
                 embedding_initializer='uniform',
                 kernel_initializer='glorot_uniform'):

    encoder_input = Input(shape=(None,), name='encoder_input')
    decoder_input = Input(shape=(None,), name='decoder_input')

    encoder_stack = encoder(num_layers, d_model, num_heads, dff,
                                input_vocab_size, pe_input_max, dropout_rate,
                                embedding_initializer=embedding_initializer,
                                kernel_initializer=kernel_initializer)

    decoder_stack = decoder(num_layers, d_model, num_heads, dff,
                                target_vocab_size, pe_target_max, dropout_rate,
                                embedding_initializer=embedding_initializer,
                                kernel_initializer=kernel_initializer)

    final_layer = Dense(target_vocab_size,
                        kernel_initializer=kernel_initializer)

    encoder_output = encoder_stack(encoder_input)
    decoder_output = decoder_stack(
        dict(decoder_input=decoder_input, encoder_output=encoder_output))
    final_output = final_layer(decoder_output)
    #delattr(final_output, '_keras_mask')

    return Model(inputs=[encoder_input, decoder_input], outputs=final_output)

