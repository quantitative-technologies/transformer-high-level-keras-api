import json
import logging
import os
# Turn off non-error logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pathlib
import pytictoc
import sys

import sacrebleu
from tqdm import tqdm, trange

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds

#import tensorflow_text
from argparse import ArgumentParser
from pytictoc import TicToc

from transformer.dataset import Dataset
from transformer.transformer import transformer
from transformer.loss import MaskedSparseCategoricalCrossentropy
from transformer.autoregression import autoregress, translate, evaluate_single
from transformer.schedule import CustomSchedule
from transformer.metrics import SequenceAccuracy, accuracy_function, correct_accuracy, bleu_scores
from prepare_tokenizers import prepare_tokenizers

base_dir = os.getcwd()
#TRAIN_DIR = os.path.join(base_dir, 'train')
TRAIN_DIR = 'train'
TOKENIZER_DIR = TRAIN_DIR
DEFAULT_MODE = 'train'
EPOCHS = 10
STEPS_PER_EPOCH = None  # Useful for debugging
BATCH_SIZE = 64
MAX_LEN = 40
NUM_HEADS = 8
DROPOUT_RATE = 0.1
INPUT_FILENAME = "inputs.pt_en.json"
#EAGERLY = False


def load_weights(model, checkpoint=None):
    """
    Load the model weights from a checkpoint.

    :param model: TensorFlow model
    :param checkpoint: Optional checkpoint to load, latest if None
    :return: Whether the checkpoint was found
    """
    if checkpoint is None:
        checkpoint = tf.train.latest_checkpoint(TRAIN_DIR)
        if checkpoint is None:
            print("No model has been trained yet.", file=sys.stderr)
            return False
    else:
        checkpoint = f"{TRAIN_DIR}/checkpoint.{checkpoint}.ckpt"

    print(f"Loading checkpoint {checkpoint}")
    model.load_weights(checkpoint).expect_partial()
    return True


def get_sequence_accuracy_metric(end_token, batch_size, max_len_target):
    seq_accuracy = SequenceAccuracy(end_token=end_token,
                                    batch_size=batch_size, max_len=max_len_target, name='SequenceAccuracy')
    return seq_accuracy


def get_compiled_model(num_layers, d_model, num_heads, dff, input_vocab_size,
                       target_vocab_size, end_token, batch_size, max_len_input,
                       max_len_target, dropout_rate,
                       embedding_initializer='uniform',
                       kernel_initializer='glorot_uniform', eagerly=False):
    """ Create the transformer_model model and compile it
    :param num_layers:
    :param d_model:
    :param num_heads:
    :param dff:
    :param input_vocab_size:
    :param target_vocab_size:
    :param end_token:
    :param batch_size:
    :param max_len_input:
    :param max_len_target:
    :param dropout_rate:
    :param embedding_initializer:
    :param kernel_initializer:
    :param eagerly:
    :return:
    """
    model = transformer(num_layers=num_layers, d_model=d_model,
                              num_heads=num_heads, dff=dff,
                              input_vocab_size=input_vocab_size.numpy(),
                              target_vocab_size=target_vocab_size.numpy(),
                              pe_input_max=max_len_input,
                              pe_target_max=max_len_target,
                              dropout_rate=dropout_rate,
                              embedding_initializer=embedding_initializer,
                              kernel_initializer=kernel_initializer)

    learning_rate = CustomSchedule(d_model=d_model)
    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    seq_accuracy = get_sequence_accuracy_metric(end_token.numpy(), batch_size,
                                                max_len_target)
    model.compile(optimizer=optimizer,
                  loss=MaskedSparseCategoricalCrossentropy(),
                  metrics=['accuracy'],
                  run_eagerly=eagerly)

    return model


def show_blue_scores(reference, prediction):
    scores = bleu_scores(reference, prediction)
    scores = [s * 100 for s in scores]
    print(
        f'BLEU scores. BLEU-1: {scores[0]:.1f}, BLEU-2: {scores[1]:.1f}, BLEU-3: {scores[2]:.1f}, BLEU-4: {scores[3]:.1f}, BLUE-4 Smoothed: {scores[4]:.1f}')


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
    # Set tensorflow logging level (no warnings)
    #tf.get_logger().setLevel('ERROR')
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = ArgumentParser()
    # parser.add_argument(
    #     '--dataset',
    #     default=DATA_SET,
    #     help="Dataset: currenty pt_to_en or abstract (default: {})".format(DATA_SET)
    # )
    parser.add_argument(
        '--mode',
        default=DEFAULT_MODE,
        help="Operation mode: train, evaluate or input(default: {})".format(DEFAULT_MODE)
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help="Number of epoch to train (default: {})".format(EPOCHS)
    )
    parser.add_argument(
        '--max-len',
        type=int,
        default=MAX_LEN,
        help=f"Maximum length of both the tokenized input/target sequences (default: {MAX_LEN})"
    )
    parser.add_argument(
        '--max-len-input',
        type=int,
        default=None,
        help=f"Maximum length of the tokenized input sequences (default: {None})"
    )
    parser.add_argument(
        '--max-len-target',
        type=int,
        default=None,
        help=f"Maximum length of the tokenized target sequences (default: {None})"
    )
    parser.add_argument(
        '-c', '--checkpoint',
        default=None,
        help=f"Model checkpoint to load, latest if None (default: {None})"
    )
    parser.add_argument(
        '--heads',
        type=int,
        default=NUM_HEADS,
        help=f"Number of heads in multihead attention layers (default: {NUM_HEADS})"
    )
    parser.add_argument(
        '-d', '--dropout',
        type=float,
        default=DROPOUT_RATE,
        help=f"The dropout rate during training (default: {DROPOUT_RATE})"
    )
    parser.add_argument(
        '-f', '--input-filename',
        default=INPUT_FILENAME,
        help=f"Filename containing inputs to predict in input mode (default: {INPUT_FILENAME})"
    )
    parser.add_argument(
        '--show-bleu',
        action='store_true',
        help=f"Show BLEU sentence scores for each test example (default: {False})"
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help=f"Remove all training checkpoints, if any (default: {False})"
    )
    parser.add_argument(
        '--clear-all',
        action='store_true',
        help=f"Delete tokenizers if any, as well as checkpoints (default: {False})"
    )
    parser.add_argument(
        '--eagerly',
        action='store_true',
        help=f"Run tensorflow in eager mode (default: {False})"
    )
    flags = parser.parse_args()

    # clean up if requested
    if flags.clear or flags.clear_all:
        checkpoint_paths = tf.io.gfile.glob(f'{TRAIN_DIR}/checkpoint*')
        for path in checkpoint_paths:
            tf.io.gfile.remove(path)
    reuse_tokenizers = not flags.clear_all

    # set maximum sequence lengths
    max_len_input = flags.max_len_input if flags.max_len_input is not None else flags.max_len
    max_len_target = flags.max_len_target if flags.max_len_target is not None else flags.max_len

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    keys = metadata.supervised_keys
    train_examples, eval_examples = examples['train'], examples['validation']

    tokenizers = prepare_tokenizers(train_examples,
                                    lower_case=True,
                                    input_vocab_size=2 ** 13,
                                    target_vocab_size=2 ** 13,
                                    name=metadata.name + '-' + keys[0] + '_to_' + keys[1],
                                    tokenizer_dir=TOKENIZER_DIR,
                                    reuse=reuse_tokenizers)

    # dataset = DatasetAdaptor(train_examples,
    #                          eval_examples,
    #                          tokenizers=tokenizers,
    #                          target_vocab_size=2 ** 13,
    #                          name=metadata.name,
    #                          keys=metadata.supervised_keys)

    dataset = Dataset(tokenizers, batch_size=BATCH_SIZE,
                      input_seqlen=max_len_input, target_seqlen=max_len_target)

    input_vocab_size = tokenizers.inputs.get_vocab_size()
    target_vocab_size = tokenizers.targets.get_vocab_size()
    print("Number of input tokens: {}".format(input_vocab_size))
    print("Number of target tokens: {}".format(target_vocab_size))

    # def tokenize_pairs3(x, y):
    #     inputs = tokenizers.inputs.tokenize(x) #.to_tensor()
    #     targets = tokenizers.targets.tokenize(y) #.to_tensor()
    #
    #     decoder_inputs = targets[:-1]
    #     decoder_targets = targets[1:]
    #     return dict(encoder_input=inputs, decoder_input=decoder_inputs), decoder_targets
    #
    # def tokenize_pairs4(x, y):
    #     return tokenize_pairs3([x], [y])
    #
    # def filter_max_length(max_x_length, max_y_length):
    #     def filter(x, y):
    #         return tf.logical_and(tf.size(x['encoder_input']) <= max_x_length,
    #                               tf.size(y) < max_y_length)
    #     return filter

    #BUFFER_SIZE = 20000
    #BATCH_SIZE = 64


    # def make_batches(ds):
    #     return (
    #         ds
    #             #.cache()
    #             .map(tokenize_pairs4)
    #             .filter(filter_max_length(max_x_length=MAX_LEN, max_y_length=MAX_LEN))
    #             # .shuffle(BUFFER_SIZE)
    #             .padded_batch(BATCH_SIZE)
    #             .cache()
    #             .prefetch(tf.data.AUTOTUNE))
    #
    #
    # def make_batches2(ds):
    #     return (
    #         ds
    #             .cache()
    #             .map(tokenize_pairs4)
    #             .filter(filter_max_length(max_x_length=MAX_LEN, max_y_length=MAX_LEN))
    #             # .shuffle(BUFFER_SIZE)
    #             #.padded_batch(BATCH_SIZE)
    #             .prefetch(tf.data.AUTOTUNE))


    #data_train = train_examples.map(tokenize_pairs4)

    #data_train = make_batches2(train_examples)

    data_train = dataset.data_pipeline(train_examples,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #data_train = data_train.map(tokenize_pairs3, num_parallel_calls=6)

    #data_eval = make_batches(eval_examples)
    data_eval = dataset.data_pipeline(eval_examples,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #data_train = dataset.data_pipeline(is_training=True, batch_size=BATCH_SIZE, input_seqlen=MAX_LEN, target_seqlen=MAX_LEN)
    #data_eval = dataset.data_pipeline(is_training=False, batch_size=BATCH_SIZE, input_seqlen=MAX_LEN, target_seqlen=MAX_LEN)
    [end_index] = tokenizers.targets.get_reserved_token_indices('end')
    seq_accuracy_metric = get_sequence_accuracy_metric(end_token=end_index,
                                                       batch_size=BATCH_SIZE,
                                                       max_len_target=MAX_LEN)

    transformer_model = get_compiled_model(num_layers=4, d_model=128,
                                           num_heads=flags.heads, dff=512,
                                           input_vocab_size=input_vocab_size,
                                           target_vocab_size=target_vocab_size,
                                           end_token=end_index,
                                           batch_size=BATCH_SIZE,
                                           max_len_input=max_len_input,
                                           max_len_target=max_len_target,
                                           dropout_rate=flags.dropout,
                                           #embedding_initializer='identity',
                                           #kernel_initializer='identity',
                                           eagerly=flags.eagerly)

    if flags.mode == 'train':
        load_weights(transformer_model)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            TRAIN_DIR + '/checkpoint.{epoch}.ckpt',
            save_weights_only=True)

        transformer_model.fit(data_train, epochs=flags.epochs, steps_per_epoch=STEPS_PER_EPOCH, validation_data=data_eval, callbacks=model_checkpoint_callback)
    elif flags.mode == 'evaluate':
        if not load_weights(transformer_model):
            print("No model trained yet", file=sys.error)
            sys.exit(1)

        if flags.show_bleu:
            for x, y in iter(eval_examples):
                x_translated = translate(transformer_model, x.numpy(),
                                         tokenizers, MAX_LEN)
                # Cannot translate sentences longer than flags.max_len
                if x_translated is None:
                    continue
                print(f"\nInput: {x.numpy()}")
                print(f"Prediction: {x_translated}")
                print(f"Ground truth: {y}")
                #show_blue_scores(y.numpy().decode('utf-8'), x_translated)
                print(sacrebleu.sentence_bleu([y.numpy().decode('utf-8')],
                                              [x_translated]))
        #  references = [y for _, y in iter(eval_examples)]

        #i = 0
        # for x, y in iter(eval_examples):
        #     i = i + 1
        #     prediction = translate(transformer_model, x.numpy(), tokenizers, MAX_LEN)
        #     print(f"{i}th prediction: {prediction}")
        else:
            total = len(eval_examples)
            references = list()
            predictions = list()
            for x, y in tqdm(iter(eval_examples), total=total):
                z = translate(transformer_model, x.numpy(), tokenizers, MAX_LEN)
                if z is not None:
                    references.append(y.numpy().decode('utf-8'))
                    predictions.append(z)
                # i = i + 1
                # if i == 5:
                #     break

            print(sacrebleu.corpus_bleu(predictions, [references]))



            #predictions = [translate(transformer_model, x.numpy(), tokenizers, MAX_LEN) for x, _ in iter(eval_examples)]

            # scores = bleu_scores(y.numpy().decode('utf-8'), x_translated)
            # scores = [s * 100 for s in scores]
            # print(f'BLEU scores. BLEU-1: {scores[0]:.1f}, BLEU-2: {scores[1]:.1f}, BLEU-3: {scores[2]:.1f}, BLEU-4: {scores[3]:.1f}, BLUE-4 Smoothed: {scores[4]:.1f}')

            # if i >= 17:
            #     break
            # i = i + 1

    elif flags.mode == 'input':
        if not load_weights(transformer_model, flags.checkpoint):
            print("No model trained yet", file=sys.stderr)
            sys.exit(1)

        with open(flags.input_filename) as f:
            data = json.load(f)

        for example in data['examples']:
            translation = translate(transformer_model, example['input'], tokenizers, MAX_LEN)
            print(f"\nInput: {example['input']}")
            print(f"Prediction: {translation}")
            print(f"Ground truth: {example['target']}")
            if flags.show_bleu:
                print(sacrebleu.sentence_bleu([example['target']], [translation]))
                #show_blue_scores(example['target'], translation)

        #del transformer_model
        sys.exit(0)

        example_input = "tinham comido peixe com batatas fritas ?"
        translation = translate(transformer_model, example_input, tokenizers, MAX_LEN)
        print("Prediction: {}".format(translation))

