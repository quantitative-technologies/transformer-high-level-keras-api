import tensorflow as tf
import tensorflow_datasets as tfds

TOKENIZER_DIR = 'train'
BATCH_SIZE = 64
MAX_LEN = 40

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
                               
keys = metadata.supervised_keys
train_examples, eval_examples = examples['train'], examples['validation']

print(f'Keys: {metadata.supervised_keys}')

example1 = next(iter(train_examples))
print(f'Example 1: {example1}')

from prepare_tokenizers import prepare_tokenizers

tokenizers = prepare_tokenizers(train_examples,
                                lower_case=True,
                                input_vocab_size=2 ** 13,
                                target_vocab_size=2 ** 13,
                                name=metadata.name + '-' + keys[0] + '_to_' + keys[1],
                                tokenizer_dir=TOKENIZER_DIR,
                                reuse=True)

print(f"Number of input tokens: {tokenizers.inputs.get_vocab_size()}")
print(f"Number of target tokens: {tokenizers.targets.get_vocab_size()}")

example1_en_string = example1[1].numpy().decode('utf-8')
print(f'Sentence: {example1_en_string}')
tokens = tokenizers.targets.tokenize([example1_en_string])
print(f'Tokenized sentence: {tokens}')
text_tokens_example1_en = tokenizers.targets.lookup(tokens)
print(f'Text tokens: {text_tokens_example1_en}')
round_trip = tokenizers.targets.detokenize(tokens)
print(f"Convert tokens back to original sentence: {round_trip.numpy()[0][0].decode('utf-8')}")