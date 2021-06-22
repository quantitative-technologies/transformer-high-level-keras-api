import tensorflow as tf
from transformer.embedding import ScaledEmbedding


raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding="post"
)

embedding = ScaledEmbedding(input_dim=5000, output_dim=16,
                            dropout_rate=0.1, max_seqlen=40)

ouput = embedding(padded_inputs)

"""
Build a model
"""

inputs = tf.keras.Input(shape = (None,))
outputs = embedding(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

pass

pass
