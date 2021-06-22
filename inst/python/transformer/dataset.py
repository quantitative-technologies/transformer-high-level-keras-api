import tensorflow as tf

BUFFER_SIZE = 20000


class Dataset:
    """
    Provides a data pipeline suitable for use with transformers
    """
    def __init__(self, tokenizers, batch_size, input_seqlen, target_seqlen):
        self.tokenizers = tokenizers
        self.batch_size = batch_size
        self.input_seqlen = input_seqlen
        self.target_seqlen = target_seqlen

    def data_pipeline(self, examples, num_parallel_calls=None):
        return (
            examples
                .cache()
                .map(tokenize_pairs(self.tokenizers),
                     num_parallel_calls=num_parallel_calls)
                .filter(filter_max_length(max_x_length=self.input_seqlen,
                                          max_y_length=self.target_seqlen))
                .shuffle(BUFFER_SIZE)
                .padded_batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE)
        )


def filter_max_length(max_x_length, max_y_length):
    def filter(x, y):
        return tf.logical_and(tf.size(x['encoder_input']) <= max_x_length,
                              tf.size(y) < max_y_length)

    return filter


def tokenize_pairs(tokenizers):
    def tokenize(x, y):
        inputs = tokenizers.inputs.tokenize([x])[0]
        targets = tokenizers.targets.tokenize([y])[0]

        decoder_inputs = targets[:-1]
        decoder_targets = targets[1:]
        return dict(encoder_input=inputs, decoder_input=decoder_inputs), decoder_targets

    return tokenize
