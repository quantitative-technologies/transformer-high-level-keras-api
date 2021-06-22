import tensorflow_datasets as tfds

from tokenizer.subword_tokenizer import SubwordTokenizer, write_vocab_file
#from tensorflow_text_vocabulary_generation import bert_vocab_from_dataset as bert_vocab
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_TOKEN = '[START]'
END_TOKEN = '[END]'

#tokenizer = SubwordTokenizer(PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN, lower_case=True, vocab_path='en_vocab.txt')

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=[PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN],
    # Arguments for `text.BertTokenizer`
    # bert_tokenizer_params=dict(self._bert_tokenizer_params, unknown_token=self._unknown_token),
    bert_tokenizer_params=dict(lower_case=True),
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)
corpus = train_examples.map(lambda pt, en: en)
vocab = bert_vocab.bert_vocab_from_dataset(
    corpus.batch(1000).prefetch(2), **bert_vocab_args)
write_vocab_file('en_vocab.txt', vocab)


tokenizer = SubwordTokenizer(PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN, lower_case=True)
tokenizer.build_from_corpus(train_examples.map(lambda pt, en: en), vocab_size=8000)
tokenizer.save('train/ted_hrlr_translate_pt_en.en')

print(tokenizer.vocab[:10])

tokens = tokenizer.tokenize(['Hello TensorFlow!'])
tokens.numpy()
text_tokens = tokenizer.lookup(tokens)
text_tokens
round_trip = tokenizer.detokenize(tokens)


pass
