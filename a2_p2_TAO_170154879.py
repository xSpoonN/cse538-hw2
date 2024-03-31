import sys
import torch
from datasets import load_dataset
from transformers import AutoModelWithLMHead, AutoTokenizer
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2', unk_token='<unk>')
model = AutoModelWithLMHead.from_pretrained('distilbert/distilgpt2').to(device)

predictions = []
labels = []

# Iterate through the dataset
for sample in boolq_dataset['validation']:
    label = sample["answer"]

    prompt = f"{sample["passage"]}\n{sample["question"]}?\n"
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=1024, return_tensors="pt").to(device)
    #input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    output = model(input_ids)[0]

    yes_token_id = tokenizer.encode("yes")[0]
    no_token_id = tokenizer.encode("no")[0]
    yes_prob = output[0, -1, yes_token_id].item()
    no_prob = output[0, -1, no_token_id].item()

    prediction = True if yes_prob > no_prob else False
    predictions.append(prediction)
    labels.append(label)

overall_acc = accuracy_score(labels, predictions)
overall_f1 = f1_score(labels, predictions, average="macro")

yes_prec = precision_score(labels, predictions, pos_label=True)
yes_rec = recall_score(labels, predictions, pos_label=True)
yes_f1 = f1_score(labels, predictions, pos_label=True)

no_prec = precision_score(labels, predictions, pos_label=False)
no_rec = recall_score(labels, predictions, pos_label=False)
no_f1 = f1_score(labels, predictions, pos_label=False)

# Print the results
print(f"Overall: acc: {overall_acc:.3f}, f1: {overall_f1:.3f}")
print(f"Yes: prec: {yes_prec:.3f}, rec: {yes_rec:.3f}, f1: {yes_f1:.3f}")
print(f"No: prec: {no_prec:.3f}, rec: {no_rec:.3f}, f1: {no_f1:.3f}")