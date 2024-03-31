import sys
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from collections import defaultdict

boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained('distilbert/distilgpt2', unk_token='<unk>')

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
        self.bigramCounts = defaultdict(int)
        self.trigramCounts = defaultdict(int)
        self.unigramProbs = {}
        self.trigramProbs = {}

    def train(self, datasets: list[list[dict]]) -> None:
        for dataset in datasets:
            for entry in dataset['train']:
                tokens = tokenizeEntry(entry)
                for i in range(len(tokens)):
                    self.unigramCounts[tokens[i]] += 1
                    self.vocab.add(tokens[i])
                    if i < len(tokens) - 1:
                        self.bigramCounts[(tokens[i], tokens[i+1])] += 1
                        # if tokens[i] == 'Ġconfidence':
                        #     print((tokens[i], tokens[i+1]), self.bigramCounts[(tokens[i], tokens[i+1])])
                    if i < len(tokens) - 2:
                        self.trigramCounts[(tokens[i], tokens[i+1], tokens[i+2])] += 1
                        # if tokens[i] == 'Ġconfidence' or tokens[i+1] == 'Ġconfidence':
                        #     print((tokens[i], tokens[i+1], tokens[i+2]), self.trigramCounts[(tokens[i], tokens[i+1], tokens[i+2])])
        self.vocab.add('<unk>')
        self.totalUnigrams = sum(self.unigramCounts.values())
        self.totalBigrams = sum(self.bigramCounts.values())
        self.totalTrigrams = sum(self.trigramCounts.values())

        # print(f'Vocab length: {len(self.vocab)}, Total unigrams: {self.totalUnigrams}, Total bigrams: {self.totalBigrams}, Total trigrams: {self.totalTrigrams}')
        for token, count in self.unigramCounts.items():
            self.unigramProbs[token] = (count + 1) / (self.totalUnigrams + len(self.vocab))
        for trigram, count in self.trigramCounts.items():
            history = trigram[:2]
            bigramCount = self.bigramCounts.get(history, 0)
            self.trigramProbs[trigram] = (count + 1) / (bigramCount + len(self.vocab))        
        #     if trigram[0] == 'Ġas' and trigram[1] == 'Ġconfidence':
        #         print(trigram, '         ', count, '         ', bigramCount, '         ', self.trigramProbs[trigram])
        # print([(trigram, self.trigramProbs[trigram]) for trigram in self.trigramProbs if trigram[0] == 'Ġas' and trigram[1] == 'Ġconfidence']) 



    def nextProb(self, history_toks: list[str], next_toks: list[str]) -> dict[str, float]:
        history = tuple(history_toks[-2:])
        probs = {}
        for next_tok in next_toks:
            trigramProb = self.trigramProbs.get((history + (next_tok,)), 0)
            unigramProb = self.unigramProbs.get(next_tok, 0)
            probs[next_tok] = (trigramProb + unigramProb) / 2
            # print(f'trigramProb: {trigramProb}, unigramProb: {unigramProb}, {next_tok}: {probs[next_tok]:.7f}')
            # print(f'P({next_tok}|{history}) = {probs[next_tok]:.4f}')
        return probs

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print("Checkpoint 1.1")
    print(tokenizeEntry(boolq_dataset['train'][0]))
    print(tokenizeEntry(boolq_dataset['train'][-1]))
    print(tokenizeEntry(emo_dataset['train'][0]))
    print(tokenizeEntry(emo_dataset['train'][-1]))


    print("\nCheckpoint 1.2")
    lm = TrigramLM()
    lm.train([boolq_dataset, emo_dataset])

    history = ['is', 'Ġmargin', 'Ġof', 'Ġerror', 'Ġthe', 'Ġsame', 'Ġas', 'Ġconfidence']
    next = ['Ġinterval', 'Ġthe', 'Ġis']
    output = lm.nextProb(history, next)
    print(f'{'history':>10}: {history}\n{'\n'.join([f"{key:>10}: {value:.6f}" for key, value in output.items()])}\n')
    
    history = ['Ġby', 'Ġland', 'Ġor', 'Ġby']
    next = ['Ġsea', 'Ġwater','Ġcycle']
    output = lm.nextProb(history, next)
    print(f'{'history':>10}: {history}\n{'\n'.join([f"{key:>10}: {value:.6f}" for key, value in output.items()])}\n')