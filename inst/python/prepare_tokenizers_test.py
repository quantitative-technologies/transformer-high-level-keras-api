import tensorflow_datasets as tfds
from pytictoc import TicToc

from prepare_tokenizers import prepare_tokenizers

TOKENIZER_DIR = 'train'
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_TOKEN = '[START]'
END_TOKEN = '[END]'

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
keys = metadata.supervised_keys
train_examples, eval_examples = examples['train'], examples['validation']

t = TicToc()
t.tic()
tokenizers = prepare_tokenizers(train_examples,
                                #pad_token=PAD_TOKEN, unknown_token=UNKNOWN_TOKEN, start_token=START_TOKEN, end_token=END_TOKEN,
                                lower_case=True,
                                input_vocab_size=2 ** 13,
                                target_vocab_size=2 ** 13,
                                name=metadata.name + '-' + keys[0] + '_to_' + keys[1],
                                tokenizer_dir=TOKENIZER_DIR)
t.toc("Preparing tokenizers took")

tokens = tokenizers.targets.tokenize(['Hello TensorFlow!'])
print(tokens)
text_tokens = tokenizers.targets.lookup(tokens)
print(text_tokens)
pass
