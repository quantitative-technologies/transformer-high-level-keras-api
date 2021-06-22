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
from transformer.autoregression import autoregress, translate
from transformer.schedule import CustomSchedule
from prepare_tokenizers import prepare_tokenizers

#base_dir = os.getcwd()
RESOURCE = 'ted_hrlr_translate'
DATASET = 'pt_to_en'
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


def get_compiled_model(num_layers, d_model, num_heads, dff, input_vocab_size,
                       target_vocab_size, max_len_input, max_len_target,
                       dropout_rate,
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

    model.compile(optimizer=optimizer,
                  loss=MaskedSparseCategoricalCrossentropy(),
                  metrics=['accuracy'],
                  run_eagerly=eagerly)

    return model


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
    # Set tensorflow logging level (no warnings)
    #tf.get_logger().setLevel('ERROR')
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = ArgumentParser()
    parser.add_argument(
        '--resource',
        default=RESOURCE,
        help=f"The tensorflow-datasets resource name (default: {RESOURCE})"
    )
    parser.add_argument(
        '--dataset',
        default=DATASET,
        help=f"The name for the dataset with in the chosen resource (default: {DATASET})"
    )
    parser.add_argument(
        '--mode',
        default=DEFAULT_MODE,
        help=f"Operation mode: train, evaluate or input (default: {DEFAULT_MODE})"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f"Number of epoch to train (default: {EPOCHS})"
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

    examples, metadata = tfds.load(f'{flags.resource}/{flags.dataset}',
                                   with_info=True, as_supervised=True)
    keys = metadata.supervised_keys
    train_examples, eval_examples = examples['train'], examples['validation']

    tokenizers, new_toks = prepare_tokenizers(
        train_examples,
        lower_case=True,
        input_vocab_size=2 ** 13,
        target_vocab_size=2 ** 13,
        name=metadata.name + '-' + keys[0] + '_to_' + keys[1],
        tokenizer_dir=TOKENIZER_DIR,
        reuse=reuse_tokenizers)

    dataset = Dataset(tokenizers, batch_size=BATCH_SIZE,
                      input_seqlen=max_len_input, target_seqlen=max_len_target)

    input_vocab_size = tokenizers.inputs.get_vocab_size()
    target_vocab_size = tokenizers.targets.get_vocab_size()
    if new_toks:
        print("Number of input tokens: {}".format(input_vocab_size))
        print("Number of target tokens: {}".format(target_vocab_size))

    data_train = dataset.data_pipeline(train_examples,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data_eval = dataset.data_pipeline(eval_examples,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    transformer_model = get_compiled_model(num_layers=4, d_model=128,
                                           num_heads=flags.heads, dff=512,
                                           input_vocab_size=input_vocab_size,
                                           target_vocab_size=target_vocab_size,
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
        if not load_weights(transformer_model, flags.checkpoint):
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
                print(sacrebleu.sentence_bleu([y.numpy().decode('utf-8')],
                                              [x_translated]))
        else:
            total = len(eval_examples)
            references = list()
            predictions = list()
            for x, y in tqdm(iter(eval_examples), total=total):
                z = translate(transformer_model, x.numpy(), tokenizers, MAX_LEN)
                if z is not None:
                    references.append(y.numpy().decode('utf-8'))
                    predictions.append(z)

            print(sacrebleu.corpus_bleu(predictions, [references]))
    elif flags.mode == 'input':
        if not load_weights(transformer_model, flags.checkpoint):
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


