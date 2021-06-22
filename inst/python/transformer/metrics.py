import tensorflow as tf
from tensorflow.keras.metrics import Metric
import sacrebleu
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

#i = 0


# def bleu_score(real, pred_softmax):
#   real = tf.cast(real, tf.int32)
#   pred = tf.cast(tf.argmax(pred_softmax, axis=2), dtype=tf.int32)
#   return tf.py_function(func=sentence_bleu, inp=[real.ref(), pred.ref()], Tout=tf.float32)
#   #return sentence_bleu(real, pred)


def bleu_scores(reference, candidate):
    reference_tokenized = word_tokenize(reference)
    candidate_tokenized = word_tokenize(candidate)

    bleu_1 = sentence_bleu([reference_tokenized],
                           candidate_tokenized, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu([reference_tokenized],
                           candidate_tokenized, weights=(0.5, 0.5, 0, 0))
    bleu_3 = sentence_bleu(
        [reference_tokenized], candidate_tokenized, weights=(0.333, 0.333, 0.333, 0))
    bleu_4 = sentence_bleu(
        [reference_tokenized], candidate_tokenized, weights=(0.25, 0.25, 0.25, 0.25))

    bleu_4_smooth = sentence_bleu([reference_tokenized], candidate_tokenized, weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=SmoothingFunction().method3)

    SacreBLEU = sacrebleu.sentence_bleu([candidate], [reference])

    return bleu_1, bleu_2, bleu_3, bleu_4, bleu_4_smooth, SacreBLEU


def accuracy_function(real, pred_softmax):
    #accuracies = tf.math.equal(real, tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32))
    real = tf.cast(real, tf.int32)
    pred = tf.cast(tf.argmax(pred_softmax, axis=2), dtype=tf.int32)
    accuracies = tf.math.equal(real, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


class SequenceAccuracy(Metric):
    # end_token, batch_size, max_len):
    def __init__(self, name='sequence_accuracy', **kwargs):
        self._end_token = kwargs['end_token']
        #self._batch_size = kwargs['batch_size']
        self._max_len = kwargs['max_len']
        # Metric constructor complains about unknown keywords
        del kwargs['end_token']
        del kwargs['max_len']
        super(SequenceAccuracy, self).__init__(name=name, **kwargs)
        self._seq_acc = self.add_weight(name='seq_acc', initializer='zeros')
        self._count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        acc = correct_accuracy(y_true, y_pred, self._end_token, False)
        total = tf.add(tf.multiply(self._count, self._seq_acc), acc)
        self._count.assign_add(tf.constant(1.))  # , dtype=tf.float32))
        self._seq_acc.assign(tf.divide(total, self._count))
        # self._count. = self._count + tf.constant(1.) #, dtype=tf.float32)
        #self._seq_acc = tf.divide(total, self._count)

    def result(self):
        #global i
        #i = i + 1
        return self._seq_acc

    def get_config(self):
        return {'name': self.name, 'dtype': self.dtype,
                'end_token': self._end_token, 'max_len': self._max_len}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def correct_accuracy(real, pred_softmax, end_token=2, random_alteration=False):
    real = tf.cast(real, tf.int32)

    tf.debugging.Assert(tf.equal(tf.shape(real)[1], tf.shape(
        pred_softmax)[1]), data=[pred_softmax])

    pred = tf.cast(tf.argmax(pred_softmax, axis=2), tf.int32)

    mask_real = tf.math.logical_not(tf.math.equal(real, 0))

    last_token = get_first_occurrence_indices(pred, end_token)
    seq = tf.range(0, tf.shape(pred_softmax)[1])
    mask_pred = tf.math.less_equal(seq, tf.expand_dims(last_token, axis=1))

    mask = tf.math.logical_or(mask_real, mask_pred)

    correct = tf.math.equal(real, pred)
    correct = tf.cast(tf.math.logical_and(mask, correct), dtype=tf.float32)

    mask = tf.cast(mask, dtype=tf.float32)
    accuracy = tf.reduce_sum(correct) / tf.reduce_sum(mask)

    if random_alteration:
        tf.random.uniform(shape=[], minval=-1., maxval=0.)
        sample = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 1)
        accuracy = accuracy * tf.cast(sample, tf.float32)
        # if (tf.equal(samples, 0)):
        #   return 0.

    return accuracy


def get_first_occurrence_indices(sequence, eos_code):
    '''
    Returns the index of the first occurence of eos_code, or length + 1 if not present.

    Based on: https://stackoverflow.com/a/62589543/1349673

    args:
        sequence: [batch, length]
        eos_code: scalar
    '''
    #batch_size, maxlen = sequence.get_shape().as_list()
    #eos_tensor = tf.constant([[eos_code]])
    batch_size = tf.shape(sequence)[0]
    eos_column = tf.tile([[eos_code]], [batch_size, 1])
    tensor = tf.concat([sequence, tf.cast(eos_column, dtype=tf.int32)], axis=-1)
    index_all_occurrences = tf.cast(
        tf.where(tf.equal(tensor, tf.cast(eos_code, dtype=tf.int32))), tf.int32)
    index_first_occurrences = tf.math.segment_min(index_all_occurrences[:, 1],
                                                  index_all_occurrences[:, 0])

    return index_first_occurrences
