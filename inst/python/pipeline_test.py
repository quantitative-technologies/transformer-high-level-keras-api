import tensorflow as tf
import tensorflow_datasets as tfds
import time

from prepare_tokenizers import prepare_tokenizers
from transformer.dataset import Dataset

TOKENIZER_DIR = 'train'
BATCH_SIZE = 64
MAX_LEN = 40


def benchmark(dataset, num_epochs=1):
    i = 0
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            i = i + 1
            # Performing a training step
            time.sleep(0.000001)
    tf.print("Execution time:", time.perf_counter() - start_time)


examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)

train_examples, eval_examples = examples['train'], examples['validation']

keys = metadata.supervised_keys
tokenizers = prepare_tokenizers(train_examples,
                                lower_case=True,
                                input_vocab_size=2 ** 13,
                                target_vocab_size=2 ** 13,
                                name=metadata.name + '-' + keys[0] + '_to_' +
                                     keys[1],
                                tokenizer_dir=TOKENIZER_DIR,
                                reuse=True)

dataset = Dataset(tokenizers, batch_size=BATCH_SIZE, input_seqlen=MAX_LEN,
                  target_seqlen=MAX_LEN)

pipeline = dataset.data_pipeline(train_examples)

benchmark(pipeline)


def tokenize_pairs(x, y):
    inputs = tokenizers.inputs.tokenize([x])
    targets = tokenizers.targets.tokenize([y])

    decoder_inputs = targets[:-1]
    decoder_targets = targets[1:]
    return dict(encoder_input=inputs, decoder_input=decoder_inputs), decoder_targets


def filter_max_length(max_x_length, max_y_length):
    def filter(x, y):
        return tf.logical_and(tf.size(x['encoder_input']) <= max_x_length,
                              tf.size(y) < max_y_length)

    return filter


pipeline2 = (
    train_examples
        .map(tokenize_pairs)  #, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .filter(filter_max_length(MAX_LEN, MAX_LEN))
        .padded_batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
)

benchmark(pipeline2)

pipeline3 = (
    train_examples
        .map(tokenize_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .filter(filter_max_length(MAX_LEN, MAX_LEN))
        .padded_batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
)

benchmark(pipeline3)

pipeline4 = dataset.data_pipeline(train_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)

benchmark(pipeline4)

# def tokenized_text_generator(input_text, target_text):
#     tokenized_inputs = tokenizers['inputs'].encode(
#         input_text.numpy(), delimit=True)
#     tokenized_targets = tokenizers['targets'].encode(
#         target_text.numpy(), delimit=True)
#
#     decoder_input_tokenized = tokenized_targets[:-1]
#     decoder_target_tokenized = tokenized_targets[1:]
#
#     return [tokenized_inputs, decoder_input_tokenized, decoder_target_tokenized]
#
#
# def seq2seq_mapping(input_text, target_text):
#     enc_inp, dec_inp, dec_tar = \
#         tf.py_function(func=tokenized_text_generator,
#                        inp=[input_text, target_text],
#                        Tout=[tf.int32] * 3)
#     enc_inp.set_shape([None])
#     dec_inp.set_shape([None])
#     dec_tar.set_shape([None])
#
#     return [dict(encoder_input=enc_inp, decoder_input=dec_inp), dec_tar]
#
#
# def filter_max_length(x, y):
#     return tf.logical_and(tf.size(x['encoder_input']) <= MAX_LEN, tf.size(y) <= MAX_LEN)
#
#
# #dataset = train_examples.prefetch(tf.data.experimental.AUTOTUNE)
# # dataset = train_examples.map(
# #             seq2seq_input_generator(tokenizers), num_parallel_calls=tf.data.experimental.AUTOTUNE)
# dataset = train_examples \
#     .map(seq2seq_mapping, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
#     .filter(filter_max_length) \
#     .cache() \
#     .padded_batch(BATCH_SIZE) \
#     .prefetch(tf.data.experimental.AUTOTUNE)
#
# # dataset2 = DatasetAdaptor(train_examples, eval_examples,
# #                              target_vocab_size=2 ** 13, name=metadata.name, keys=metadata.supervised_keys,
# #                              tokenizer_dir=TOKENIZER_DIR, reuse_tokenizers=True)
# #
# # data_train = dataset2.data_pipeline(is_training=True, batch_size=BATCH_SIZE, input_seqlen=MAX_LEN, target_seqlen=MAX_LEN)
#
# benchmark(dataset)
# benchmark(data_train)
