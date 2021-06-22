import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, LayerNormalization


class TransformerSubLayer(Layer):
    def __init__(self, input_sublayer, input_key=None, epsilon=1e-6,
                 dropout_rate=0.1):
        super(TransformerSubLayer, self).__init__()
 
        self.input_sublayer = input_sublayer
        self.input_key = input_key
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=epsilon)

    def call(self, inputs, training=None, mask=None):
        if self.input_key is not None:
            x = inputs[self.input_key]
        else:
            x = inputs
        sublayer_outputs = self.input_sublayer(inputs=inputs, mask=mask)
        outputs = self.dropout(sublayer_outputs, training)
        outputs += x  # Loses the mask info
        return self.layernorm(outputs)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        if self.input_key:
            return mask[self.input_key]
        return mask

