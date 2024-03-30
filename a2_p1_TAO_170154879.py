from datasets import load_dataset
boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
from transformers import PreTrainedTokenizerFast
gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained('distilbert/distilgpt2')
import sys
sys.stdout.reconfigure(encoding='utf-8')

def tokenizeWithStartStop(text):
    tokens = gpt2_tokenizer.tokenize(text)
    tokens = ['<s>'] + tokens + ['</s>']
    return tokens

def tokenizeEntry(entry):
    return tokenizeWithStartStop(entry['passage'] + ' ' + entry['question'] + '? ' + str(entry['answer']))

tokenizeWithStartStop("Hello, my name is John.")
firstEntry = boolq_dataset['train'][0]
lastEntry = boolq_dataset['train'][-1]
print(firstEntry)
print(tokenizeEntry(firstEntry))
print(lastEntry)
print(tokenizeEntry(lastEntry))