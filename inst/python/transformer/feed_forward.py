from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def pointwise_feed_forward_network(d_model, dff, kernel_initializer='glorot_uniform'):
    return Sequential([
        Dense(dff, activation='relu', kernel_initializer=kernel_initializer),  # (batch_size, seq_len, dff)
        Dense(d_model, kernel_initializer=kernel_initializer)  # (batch_size, seq_len, d_model)
    ])
