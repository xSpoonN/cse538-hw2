import sys
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from collections import defaultdict

boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained('distilbert/distilgpt2')

def tokenizeWithStartStop(text: str) -> list[str]:
    tokens: list[str] = gpt2_tokenizer.tokenize(text)
    tokens = ['<s>'] + tokens + ['</s>']
    return tokens

def tokenizeEntry(entry: dict) -> list[str]:
    try:
        return tokenizeWithStartStop(entry['passage'] + ' ' + entry['question'] + '? ' + str(entry['answer']))
    except KeyError:
        return tokenizeWithStartStop(entry['text'])

class TrigramLM:
    def __init__(self):
        self.vocab = set()
        self.unigramCounts = defaultdict(int)
        self.bigramCounts = defaultdict(lambda: defaultdict(int))
        self.trigramCounts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.tokenCount = 0
        self.unigramProbs = defaultdict(float)
        self.trigramProbs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def train(self, datasets: list[list[dict]]) -> None:
        for dataset in datasets:
            for entry in dataset['train']:
                tokens = tokenizeEntry(entry)
                for i in range(2, len(tokens)):
                    self.vocab.add(tokens[i])
                    self.vocab.add(tokens[i-1])
                    self.vocab.add(tokens[i-2])

                    self.unigramCounts[tokens[i]] += 1
                    self.bigramCounts[tokens[i-1]][tokens[i]] += 1
                    self.trigramCounts[tokens[i-2]][tokens[i-1]][tokens[i]] += 1
                    self.tokenCount += 1
        print(self.bigramCounts)

        self.vocab.add('OOV')
        for token in self.vocab:
            self.unigramProbs[token] = (self.unigramCounts[token] + 1) / (self.tokenCount + len(self.vocab))

        for w2 in self.trigramCounts:
            for w1 in self.trigramCounts[w2]:
                bigramCount = sum(self.trigramCounts[w2][w1].values())
                for w0 in self.trigramCounts[w2][w1]:
                    self.trigramProbs[w2][w1][w0] = (self.trigramCounts[w2][w1][w0] + 1) / (bigramCount + len(self.vocab))

    def nextProb(self, history_toks: list[str], next_toks: list[str]) -> dict[str, float]:
        if len(history_toks) < 2:
            history_toks  = ['<s>', '</s>'] + history_toks
        else:
            history_toks = history_toks[-2:]
        probs = {}
        for next_tok in next_toks:
            unigramProb = self.unigramProbs[next_tok]
            trigramProb = self.trigramProbs[history_toks[0]][history_toks[1]].get(next_tok, 0)
            probs[next_tok] = (unigramProb + trigramProb) / 2
            print(f'P({next_tok}|{history_toks[0]} {history_toks[1]}) = {probs[next_tok]}')
        return probs

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print("Checkpoint 1.1")
    print(tokenizeEntry(boolq_dataset['train'][0]))
    print(tokenizeEntry(boolq_dataset['train'][-1]))
    print(tokenizeEntry(emo_dataset['train'][0]))
    print(tokenizeEntry(emo_dataset['train'][-1]))


    print("Checkpoint 1.2")
    lm = TrigramLM()
    lm.train([boolq_dataset, emo_dataset])
    print(lm.nextProb([
        'is', 'Ġmargin', 'Ġof', 'Ġerror', 'Ġthe', 'Ġsame', 'Ġas', 'Ġconfidence'
    ],[
        'Ġinterval', 'Ġthe', 'Ġis'
    ]))
    print(lm.nextProb([
        'Ġby', 'Ġland', 'Ġor', 'Ġby'
    ], [
        'Ġsea', 'Ġwater','Ġcycle'
    ]))