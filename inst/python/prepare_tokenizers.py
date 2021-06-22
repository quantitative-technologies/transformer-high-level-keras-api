from pathlib import Path
import tensorflow as tf
from pytictoc import TicToc

from tokenizer.subword_tokenizer import SubwordTokenizer

_PAD_TOKEN = '[PAD]'
_UNKNOWN_TOKEN = '[UNK]'
_START_TOKEN = '[START]'
_END_TOKEN = '[END]'


def prepare_tokenizers(corpus_pair,
                       input_vocab_size,
                       target_vocab_size,
                       name,
                       tokenizer_dir,
                       pad_token=_PAD_TOKEN,
                       unknown_token=_UNKNOWN_TOKEN,
                       start_token=_START_TOKEN,
                       end_token=_END_TOKEN,
                       reuse=True,
                       **bert_tokenizer_params):
    """
    Prepare a pair of tokenizers from input and target corpora

    :param corpus_pair: Iterable pairs (input, target)
    :param input_vocab_size: Maximum size of the input vocab
    :param target_vocab_size: Maximum size of the target vocab
    :param name: Name for this pair
    :param tokenizer_dir: Directory to save the tokenizers
    :param pad_token: Pad token, default: _PAD_TOKEN
    :param unknown_token: Unknown token, default: _UNKNOWN_TOKEN
    :param start_token: Start token, default: _START_TOKEN
    :param end_token: End token, default: _END_TOKEN
    :param reuse: If pair according to name is already saved, load and return it, when reuse. Otherwise will force the
        creation of new tokenizers.
    :param bert_tokenizer_params:
    :return: The pair of tokenizers, and whether they were just created
    """
    p_tokenizer_dir = Path(tokenizer_dir)
    model_name = (p_tokenizer_dir / name).as_posix()

    create_tokenizers = not reuse
    if not create_tokenizers:
        try:
            tokenizers = tf.saved_model.load(model_name)
        except OSError as e:
            create_tokenizers = True

    if create_tokenizers:
        t = TicToc()
        tf.print('Creating tokenizers. Please be patient...')
        t.tic()
        tokenizer_inputs = SubwordTokenizer(pad_token, unknown_token, start_token, end_token, **bert_tokenizer_params)
        tokenizer_inputs.build_from_corpus(corpus_pair.map(lambda x, _: x), vocab_size=input_vocab_size)
        tokenizer_targets = SubwordTokenizer(pad_token, unknown_token, start_token, end_token, **bert_tokenizer_params)
        tokenizer_targets.build_from_corpus(corpus_pair.map(lambda _, y: y), vocab_size=target_vocab_size)
        tokenizers = tf.Module()
        tokenizers.inputs = tokenizer_inputs
        tokenizers.targets = tokenizer_targets
        tf.saved_model.save(tokenizers, model_name)
        t.toc('Creating tokenizers took')

    return tokenizers, create_tokenizers