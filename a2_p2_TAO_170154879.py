import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from a2_p1_TAO_170154879 import *
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2', unk_token='<unk>')
gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained('distilbert/distilgpt2', unk_token='<unk>')
model = AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2').to(device)

predictions = []
labels = []

yes_token_id = tokenizer.encode("yes")[0]
no_token_id = tokenizer.encode("no")[0]

for sample in boolq_dataset['validation']:
    label = sample["answer"]

    prompt = f"{sample["passage"]}\n{sample["question"]}?\n"
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=1024, return_tensors="pt").to(device)

    output = model(input_ids)[0]

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

print("Checkpoint 2.1")
print(f"Overall: acc: {overall_acc:.3f}, f1: {overall_f1:.3f}")
print(f"Yes: prec: {yes_prec:.3f}, rec: {yes_rec:.3f}, f1: {yes_f1:.3f}")
print(f"No: prec: {no_prec:.3f}, rec: {no_rec:.3f}, f1: {no_f1:.3f}")

lm = TrigramLM()
lm.train([boolq_dataset, emo_dataset])

predictions = []
labels = []
for sample in boolq_dataset['validation']:
    label = sample["answer"]

    history = tokenizeEntry({"passage": sample["passage"], "question": sample["question"], "answer": ''})
    next_toks = ['yes', 'no']
    probs = lm.nextProb(history, next_toks)
    prediction = True if probs['yes'] > probs['no'] else False
    predictions.append(prediction)
    labels.append(label)

overall_acc = accuracy_score(labels, predictions)
overall_f1 = f1_score(labels, predictions, average='macro', zero_division=0)

yes_prec = precision_score(labels, predictions, pos_label=True, zero_division=0)
yes_rec = recall_score(labels, predictions, pos_label=True, zero_division=0)
yes_f1 = f1_score(labels, predictions, pos_label=True, zero_division=0)

no_prec = precision_score(labels, predictions, pos_label=False, zero_division=0)
no_rec = recall_score(labels, predictions, pos_label=False, zero_division=0)
no_f1 = f1_score(labels, predictions, pos_label=False, zero_division=0)

print("\nCheckpoint 2.2")
print(f"Overall: acc: {overall_acc:.3f}, f1: {overall_f1:.3f}")
print(f"Yes: prec: {yes_prec:.3f}, rec: {yes_rec:.3f}, f1: {yes_f1:.3f}")
print(f"No: prec: {no_prec:.3f}, rec: {no_rec:.3f}, f1: {no_f1:.3f}")

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2', unk_token='<unk>')
numEpochs = 8
learningRate = 1e-4
weightDecay = 0.001

train_losses = []
model = AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2').to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weightDecay)
for epoch in range(numEpochs):
    model.train()
    totalLoss = 0
    print(f"Epoch {epoch + 1}/{numEpochs}")
    for i, batch in enumerate(boolq_dataset['train']):
        if i % 500 == 0:
            print(f'Batch {i}/{len(boolq_dataset["train"])}')
        optimizer.zero_grad()
        passage = batch['passage']
        question = batch['question']
        label = batch['answer']
        input_text = f"{passage}\n{question}?\n{label}"
        #print(input_texts)
        inputs = tokenizer(input_text, truncation=True, max_length=1024, return_tensors="pt").to(device)
        #print(inputs)
        #print(inputs["input_ids"])

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()
    train_losses.append(totalLoss / len(boolq_dataset['train']))
plt.figure(figsize=(10, 6))
plt.plot(range(1, numEpochs + 1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("training_loss.png")

model.eval()
predictions = []
labels = []

yes_token_id = tokenizer.encode("yes")[0]
no_token_id = tokenizer.encode("no")[0]

for sample in boolq_dataset['validation']:
    label = sample["answer"]

    prompt = f"{sample["passage"]}\n{sample["question"]}?\n"
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=1024, return_tensors="pt").to(device)

    output = model(input_ids)[0]

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

print("\nCheckpoint 2.4")
print(f"Overall: acc: {overall_acc:.3f}, f1: {overall_f1:.3f}")
print(f"Yes: prec: {yes_prec:.3f}, rec: {yes_rec:.3f}, f1: {yes_f1:.3f}")
print(f"No: prec: {no_prec:.3f}, rec: {no_rec:.3f}, f1: {no_f1:.3f}")