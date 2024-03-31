import sys
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from collections import defaultdict

boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained('distilbert/distilgpt2', unk_token='<unk>')

with open('testout.txt', 'w', encoding='utf-8') as f:
    for example in emo_dataset['train']:
        f.write(str(example) + '\n')
        tokens = gpt2_tokenizer.tokenize(example['text'])
        f.write(' '.join(tokens) + '\n')
        f.write('\n')