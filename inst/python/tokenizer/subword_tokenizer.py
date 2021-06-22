
import pathlib
import re
import tensorflow as tf
#from tensorflow_text import text
from tensorflow_text import BertTokenizer


# NOTE: With tensorflow_text version 2.5 the
# tensorflow_text_vocabulary_generation module will not be needed. Can use:
#
#from tensorflow_text.tools.bert_vocab_from_dataset import bert_vocab_from_dataset as bert_vocab
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
#
# instead.

#from tensorflow_text_vocabulary_generation import bert_vocab_from_dataset as bert_vocab


class SubwordTokenizer(tf.Module):
    """
    Subword tokenizer

    Based on CustomTokenizer from https://github.com/tensorflow/text/blob/master/examples/subwords_tokenizer.ipynb,
    which in turn uses the BertTokenizer
    """

    def __init__(self, pad_token, unknown_token, start_token, end_token,
                 additional_reserved_tokens=[], vocab_path=None,
                 **bert_tokenizer_params):
        """
        :param pad_token:
        :param unknown_token:
        :param start_token:
        :param end_token:
        :param additional_reserved_tokens: Optional list of additional reserved tokens.
        :param vocab_path: Optional path to vocab file, which is used to build
            the tokenizer if provided.
        :param bert_tokenizer_params: Additional parameters,
            besides unknown_token, for BertTokenizer.
        """

        self._start_token = start_token
        self._end_token = end_token
        self._unknown_token = unknown_token
        self._reserved_tokens = [pad_token, unknown_token, start_token, end_token] + additional_reserved_tokens
        self._bert_tokenizer_params = bert_tokenizer_params
        # Indices of the special tokens
        self._pad = tf.constant(0, dtype=tf.int64)
        self._unknown = tf.constant(1, dtype=tf.int64)
        self._start = tf.constant(2, dtype=tf.int64)
        self._end = tf.constant(3, dtype=tf.int64)

        if vocab_path is None:
            self.tokenizer = None
            self.vocab = None
            self._vocab_path = None
        else:
            self.tokenizer = BertTokenizer(vocab_path, unknown_token=unknown_token,
                                           **bert_tokenizer_params)
            vocab = pathlib.Path(vocab_path).read_text().splitlines()
            self.vocab = tf.Variable(vocab)
            self._vocab_path = tf.saved_model.Asset(vocab_path)
            self._create_signatures()

    def _create_signatures(self):
        ## Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
        self.get_reserved_tokens.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.string))
        self.get_reserved_token_indices.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.string))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()

    def build_from_corpus(self, corpus, vocab_size):
        bert_vocab_args = dict(
            # The target vocabulary size
            vocab_size=vocab_size,
            # Reserved tokens that must be included in the vocabulary
            reserved_tokens=self._reserved_tokens,
            # Arguments for `text.BertTokenizer`
            #bert_tokenizer_params=dict(self._bert_tokenizer_params, unknown_token=self._unknown_token),
            bert_tokenizer_params=self._bert_tokenizer_params,
            # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
            learn_params={},
        )
        vocab = bert_vocab.bert_vocab_from_dataset(
            corpus.batch(1000).prefetch(2), **bert_vocab_args)
        self.vocab = tf.Variable(vocab)
        lookup = tf.lookup.StaticVocabularyTable(
            num_oov_buckets=1,
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=vocab,
                values=tf.range(len(vocab), dtype=tf.int64)))
        self.tokenizer = BertTokenizer(lookup, unknown_token=self._unknown_token, **self._bert_tokenizer_params)
        self._create_signatures()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = self.add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        """
        Inverts the tokenize operation.

        :param tokenized:
        :return: detokenized sequence of tokens
        """
        words = self.tokenizer.detokenize(tf.expand_dims(tokenized, axis=0))
        return self.cleanup_text(words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self, which='all'):
        """
        Get the reserved tokens

        :param which:
        :return: tensor of token(s)
        """
        if which == 'all':
            return tf.constant(self._reserved_tokens)
        elif which == 'pad':
            return tf.constant(self._reserved_tokens[self._pad])
        elif which == 'unknown':
            return tf.constant(self._reserved_tokens[self._unknown])
        elif which == 'end':
            return tf.constant(self._reserved_tokens[self._end])
        else:
            return tf.constant('')

    @tf.function
    def get_reserved_token_indices(self, which='all'):
        if which == 'all':
            return tf.range(start=0, limit=len(self._reserved_tokens), dtype=tf.int64)
        elif which == 'pad':
            return tf.range(start=self._pad, limit=self._pad + 1, dtype=tf.int64)
        elif which == 'unknown':
            return tf.range(start=self._unknown, limit=self._unknown + 1, dtype=tf.int64)
        elif which == 'end':
            return tf.range(start=self._end, limit=self._end + 1, dtype=tf.int64)
        else:
            return tf.range(start=-1, limit=0, dtype=tf.int64)

    @tf.function
    def add_start_end(self, ragged):
        count = ragged.bounding_shape()[0]
        starts = tf.fill([count, 1], self._start)
        ends = tf.fill([count, 1], self._end)
        return tf.concat([starts, ragged, ends], axis=1)

    @tf.function
    def cleanup_text(self, token_txt):
        # Drop the reserved tokens, except for "[UNK]".
        bad_tokens = [re.escape(tok) for tok in self._reserved_tokens if
                      tok != self._unknown_token]
        bad_token_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
        result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # Join them into strings.
        result = tf.strings.reduce_join(result, separator=' ', axis=-1)

        return result

    def save(self, model_name):
        """
        Save this.

        :param model_name: directory where the Module will be saved
        """
        tf.saved_model.save(self, model_name)

    @classmethod
    def load(cls, model_name):
        return tf.saved_model.load(model_name)


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)
