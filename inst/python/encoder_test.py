import tensorflow as tf
from transformer.attention import MultiHeadSelfAttention
from transformer.encoder import encoder, encoder_layer
from transformer.embedding import ScaledEmbedding
from transformer.mask import PaddingMask
from transformer.sublayer import TransformerSubLayer
from transformer.feed_forward import pointwise_feed_forward_network

raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding="post"
)

embedding = ScaledEmbedding(input_dim=5000, output_dim=128,
                            dropout_rate=0.1, max_seqlen=40)

embedded_inputs = embedding(padded_inputs)

mha = MultiHeadSelfAttention(d_model=128, num_heads=8)

out1 = mha(embedded_inputs)

sublayer = TransformerSubLayer(mha, epsilon=1e-6, dropout_rate=0.1)

out2 = sublayer(embedded_inputs)

# Test 3
embedding3 = tf.keras.Sequential([PaddingMask(),
                                  ScaledEmbedding(input_dim=5000, output_dim=128,
                                                  dropout_rate=0.0, max_seqlen=40, embedding_initializer='identity')])
embedded_inputs3 = embedding3(padded_inputs, training=True)
mha3 = MultiHeadSelfAttention(d_model=128, num_heads=8, kernel_initializer='identity')
mha_out3 = mha3(embedded_inputs3)
ffn3 = pointwise_feed_forward_network(d_model=128, dff=512, kernel_initializer='identity')
ffn_out3 = ffn3(embedded_inputs3)
enc_layer = encoder_layer(d_model=128, num_heads=8, dff=512, dropout_rate=0.0, kernel_initializer='identity')
out3 = enc_layer(embedded_inputs3, training=True)
model3 = encoder(num_layers=6, d_model=128, num_heads=8, dff=512,
                    input_vocab_size=5000, maximum_position_encoding=40,
                    dropout_rate=0.1, embedding_initializer='identity', kernel_initializer='identity')
outputs3 = model3(padded_inputs)

pass
