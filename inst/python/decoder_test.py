import tensorflow as tf
from tensorflow.keras.layers import Masking
from transformer.attention import MultiHeadSelfAttention, MultiHeadTwoInputAttention
from transformer.decoder import decoder_layer, test_layer, DecoderLayer, Decoder
from transformer.embedding import ScaledEmbedding
from transformer.sublayer import TransformerSubLayer
from transformer.mask import AutoRegressiveMask, PaddingMask, \
    create_padding_mask, create_look_ahead_mask
from transformerBlog.inst.python.transformer.decoder import decoder

raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]
raw_inputs2 = [
    [711, 632, 71, 100],
    [73, 8, 3215, 55, 927, 2],
    [83, 91, 1, 645, 1253, 927, 1777],
]

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding="post"
)
padded_inputs2 = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs2, padding="post"
)


def create_mask(inputs, mask=None):
    # decoder_input, encoder_output = inputs
    decoder_padding_mask = create_padding_mask(inputs)
    look_ahead_mask = create_look_ahead_mask(
        tf.shape(inputs)[1])
    return tf.logical_and(decoder_padding_mask, look_ahead_mask)


example1 = False
example2 = False
example3 = False
example4 = False
example5 = False
example6 = False
example7 = False
example8 = False
example9 = False
example10 = True
example11 = False
example12 = False

if example1:
    # Test AutoRegressiveMask layer
    auto_regress1 = AutoRegressiveMask()
    embedding1 = tf.keras.Sequential([
        PaddingMask(),
        ScaledEmbedding(input_dim=5000, output_dim=128, dropout_rate=0.1,
                        max_seqlen=40),
    ])
    embedded_inputs1 = embedding1(padded_inputs)
    #print(embedded_inputs1._keras_mask)
    embedded_inputs1_auto_reg = auto_regress1(embedded_inputs1)
    print(embedded_inputs1_auto_reg._keras_mask)
    print("End example 1")

if example2:
    # Test self-attention (type 2)
    auto_regress2 = AutoRegressiveMask()
    embedding2 = tf.keras.Sequential([
        PaddingMask(),
        ScaledEmbedding(input_dim=5000, output_dim=128, dropout_rate=0.1,
                        max_seqlen=40),
    ])
    mask = create_mask(padded_inputs2)
    embedded_inputs2 = embedding2(padded_inputs2)
    mha2 = MultiHeadSelfAttention(d_model=128, num_heads=8)
    mha_auto_reg2 = tf.keras.Sequential([auto_regress2, mha2])
    out2 = mha_auto_reg2(embedded_inputs2, mask=mask)
    print(out2)
    #print(out2._keras_mask)
    print("End example 2")

if example3:
    # Test self-attention (type 1)
    embedding3 = tf.keras.Sequential([
        PaddingMask(),
        ScaledEmbedding(input_dim=5000, output_dim=128, dropout_rate=0.1,
                        max_seqlen=40, embedding_initializer='identity'),
    ])
    mask = create_mask(padded_inputs2)
    embedded_inputs3 = embedding3(padded_inputs2)
    auto_regress3 = AutoRegressiveMask()
    mha3 = MultiHeadSelfAttention(d_model=128, num_heads=8, kernel_initializer='identity')
    mha3_seq = tf.keras.Sequential([auto_regress3, mha3])
    # (type 1)
    #mha3_seq = tf.keras.Sequential([mha3])
    # (working)
    #out3 = mha3(embedded_inputs3, mask=mask)
    out3 = mha3_seq(embedded_inputs3)
    #out3 = mha3_seq(embedded_inputs3, mask=mask)
    print(out3)
    print("End example 3")

if example4:
    # Test two-input attention
    embedding4enc, embedding4dec = [
        tf.keras.Sequential([PaddingMask(),
                             ScaledEmbedding(input_dim=5000, output_dim=128,
                                             dropout_rate=0.1, max_seqlen=40)]) for _ in range(2)]
    embedded_inputs4enc = embedding4enc(padded_inputs)
    embedded_inputs4dec = embedding4dec(padded_inputs2)
    mha2in = MultiHeadTwoInputAttention(d_model=128, num_heads=8)
    out4 = mha2in(inputs=dict(queries=embedded_inputs4dec, lookups=embedded_inputs4enc))
    #out4 = mha2in(inputs=dict(queries=out2, lookups=embedded_inputs4enc))
    print(out4._keras_mask)
    print("End example 4")

if example5:
    # Test two-input attention TransformerSublayer
    embedding5enc, embedding5dec = [
        tf.keras.Sequential([PaddingMask(),
                             ScaledEmbedding(input_dim=5000, output_dim=128,
                                             dropout_rate=0.1, max_seqlen=40)]) for _ in range(2)]
    embedded_inputs5enc = embedding5enc(padded_inputs)
    embedded_inputs5dec = embedding5dec(padded_inputs2)
    mha2in = MultiHeadTwoInputAttention(d_model=128, num_heads=8)
    sublayer = TransformerSubLayer(mha2in, input_key='queries', epsilon=1e-6, dropout_rate=0.1)
    out5 = sublayer(dict(lookups=embedded_inputs5enc, queries=embedded_inputs5dec))
    print(out5._keras_mask)
    print("End example 5")

if example6:
    # Test decoder_layer
    embedding6enc, embedding6dec = [
        tf.keras.Sequential([PaddingMask(),
                             ScaledEmbedding(input_dim=5000, output_dim=128,
                                             dropout_rate=0.1, max_seqlen=40)]) for _ in range(2)]
    embedded_inputs6enc = embedding6enc(padded_inputs)
    embedded_inputs6dec = embedding6dec(padded_inputs2)
    layer = decoder_layer(d_model=128, num_heads=8, dff=512, dropout_rate=0.1)
    outputs = layer(dict(encoder_output=embedded_inputs6enc, decoder_input=embedded_inputs6dec))
    print(outputs._keras_mask)
    print("End example 6")

if example7:
    # Test a sequence of 2 decoding layers
    embedding7enc, embedding7dec = [
        tf.keras.Sequential([PaddingMask(),
                             ScaledEmbedding(input_dim=5000, output_dim=128,
                                             dropout_rate=0.1, max_seqlen=40)]) for _ in range(2)]

    embedded_inputs7enc = embedding7enc(padded_inputs)
    embedded_inputs7dec = embedding7dec(padded_inputs2)
    layer1, layer2 = [decoder_layer(d_model=128, num_heads=8, dff=512, dropout_rate=0.1) for _ in range(2)]
    outputs = layer1(dict(encoder_output=embedded_inputs7enc, decoder_input=embedded_inputs7dec))
    outputs = layer2(dict(encoder_output=embedded_inputs7enc, decoder_input=outputs))
    print(outputs._keras_mask)
    print("End example 7")

if example8:
    embedding8 = tf.keras.Sequential([
        PaddingMask(),
        ScaledEmbedding(input_dim=5000, output_dim=128, dropout_rate=0.1,
                                 max_seqlen=40)
    ])
    embedded_inputs8 = embedding8(padded_inputs)
    mha_auto_reg = tf.keras.Sequential([
        AutoRegressiveMask(),
        MultiHeadSelfAttention(d_model=128, num_heads=8)
    ])
    mha_self_sublayer = tf.keras.Sequential([
        TransformerSubLayer(mha_auto_reg, epsilon=1e-6, dropout_rate=0.1)
        #PaddingMask()
    ])
    outputs = mha_self_sublayer(embedded_inputs8)
    print(outputs._keras_mask)
    print("End example 8")

if example9:
    # DecoderLayer
    embedding9enc, embedding9dec = [
        tf.keras.Sequential([PaddingMask(),
                             ScaledEmbedding(input_dim=5000, output_dim=128,
                                             dropout_rate=0.0, max_seqlen=40,
                                             embedding_initializer='identity')]) for _ in range(2)]
    embedded_inputs9enc = embedding9enc(padded_inputs, training=True)
    embedded_inputs9dec = embedding9dec(padded_inputs2, training=True)
    layer = DecoderLayer(d_model=128, num_heads=8, dff=512, dropout_rate=0.0, kernel_initializer='identity')
    mask = embedded_inputs9enc._keras_mask
    decoder_mask = Decoder.create_mask(dict(decoder_input=padded_inputs2))
    outputs = layer([embedded_inputs9dec, embedded_inputs9enc], training=True, mask=[mask, decoder_mask])
    #outputs = layer(dict(encoder_output=embedded_inputs9enc, decoder_input=embedded_inputs9dec))
    print(outputs)
    print("End example 9")

if example10:
    # Decoder
    embedding10enc = tf.keras.Sequential([PaddingMask(),
                                          ScaledEmbedding(input_dim=5000,
                                                          output_dim=128,
                                                          dropout_rate=0.0,
                                                          max_seqlen=40,
                                                          embedding_initializer='identity')])
    embedded_inputs10enc = embedding10enc(padded_inputs, training=True)
    #embedded_inputs10dec = embedding10dec(padded_inputs2, training=True)
    layer = decoder(num_layers=1, d_model=128, num_heads=8, dff=512,
                    target_vocab_size=5000, maximum_position_encoding=40,
                    dropout_rate=0.0, embedding_initializer='identity', kernel_initializer='identity')
    inputs = dict(decoder_input=padded_inputs2, encoder_output=embedded_inputs10enc)
    outputs = layer(inputs)
    #mask = embedded_inputs10enc._keras_mask
    #decoder_mask = Decoder.create_mask(dict(decoder_input=padded_inputs2))
    #outputs = layer([embedded_inputs10dec, embedded_inputs10enc], training=True, mask=[mask, decoder_mask])
    #outputs = layer(dict(encoder_output=embedded_inputs10enc, decoder_input=embedded_inputs10dec))
    print(outputs)
    print("End example 10")

if example11:
    # Test self-attention sublayer (type 1)
    embedding11 = tf.keras.Sequential([
        PaddingMask(),
        ScaledEmbedding(input_dim=5000, output_dim=128, dropout_rate=0.0,
                        max_seqlen=40, embedding_initializer='identity'),
    ])
    mask = create_mask(padded_inputs2)
    embedded_inputs11 = embedding11(padded_inputs2)
    mha11 = MultiHeadSelfAttention(d_model=128, num_heads=8, kernel_initializer='identity')
    mha11_seq = tf.keras.Sequential([mha11])
    mha11_sublayer = TransformerSubLayer(mha11_seq, epsilon=1e-6, dropout_rate=0.0)
    out11 = mha11_sublayer(embedded_inputs11, mask=mask)
    print(out11)
    print("End example 11")

if example12:
    # Test self-attention sublayer (working)
    embedding12 = tf.keras.Sequential([
        PaddingMask(),
        ScaledEmbedding(input_dim=5000, output_dim=128, dropout_rate=0.0,
                        max_seqlen=40, embedding_initializer='identity'),
    ])
    mask = create_mask(padded_inputs2)
    embedded_inputs12 = embedding12(padded_inputs2)
    mha12 = MultiHeadSelfAttention(d_model=128, num_heads=8, kernel_initializer='identity')
    #mha12_seq = tf.keras.Sequential([mha12])
    mha12_sublayer = TransformerSubLayer(mha12, epsilon=1e-6, dropout_rate=0.0)
    out12 = mha12_sublayer(embedded_inputs12, mask=mask)
    print(out12)
    print("End example 12")

