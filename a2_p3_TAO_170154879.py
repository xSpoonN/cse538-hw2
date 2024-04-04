import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import RobertaModel, RobertaTokenizerFast
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

class Classifier(torch.nn.Module):
    def __init__(self, basemodel):
        super().__init__()
        self.model = basemodel
        self.linear = torch.nn.Linear(basemodel.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        logits = self.linear(output.last_hidden_state[:, 0, :])
        logits = torch.sigmoid(logits)
        return logits
    
class Regression(torch.nn.Module):
    def __init__(self, basemodel):
        super().__init__()
        self.model = basemodel
        self.linear = torch.nn.Linear(basemodel.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        logits = self.linear(output.last_hidden_state[:, 0, :])
        return logits

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    boolq_dataset = load_dataset('google/boolq')
    emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence', trust_remote_code=True)
    model = RobertaModel.from_pretrained('distilroberta-base').to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base', pad_token='<pad>', unk_token='<unk>')

    train_dataset = boolq_dataset['train'].map(lambda example: {'label': 1 if example['answer'] == True else 0})
    validate_dataset = boolq_dataset['validation'].map(lambda example: {'label': 1 if example['answer'] == True else 0})

    classifier = Classifier(model).to(device)#.to(dtype=torch.float16)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)

    train_losses = []
    print('Training Classifier')
    for epoch in range(5):
        classifier.train()
        print(f"\nEpoch {epoch}", end=' ')
        epoch_loss = 0
        for i, batch in enumerate(DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)):
            # if i < 5: continue
            # if i > 10: exit()
            if i % 100 == 0: print(f"\nBatch {i} - {i+100} ...", end=' ')
            #print('.', end='', flush=True)
            optimizer.zero_grad()
            prompts = [f"{passage}\n{question}?\n" for passage, question in zip(batch['passage'], batch['question'])]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            # inputs = tokenizer(batch['passage'], batch['question'], padding=True, truncation=True, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)#.to(dtype=torch.float16)
            # labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=2).float().to(device)
            # print('inputs', inputs['input_ids'])
            # print('labels', labels)
            # exit()
            logits = classifier(**inputs).squeeze()#.to(dtype=torch.float16)
            # print('first', logits)


            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print('loss', loss)
            # logits = classifier(**inputs).squeeze()#.to(dtype=torch.float16)
            # print('second', logits)
        train_losses.append(epoch_loss / len(train_dataset))
        print(f"\nEpoch {epoch} loss: {epoch_loss / len(train_dataset)}")

    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('3.1.png')

    # Evaluation
    classifier.eval()
    predictions = []
    true_labels = []
    print('Evaluating Classifier', end=' ')
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(validate_dataset, batch_size=16, pin_memory=True, num_workers=4)):
            if i % 100 == 0: print(f"\nBatch {i} - {i+100} ...", end=' ')
            optimizer.zero_grad()
            prompts = [f"{passage}\n{question}?\n" for passage, question in zip(batch['passage'], batch['question'])]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = classifier(**inputs).squeeze()

            predictions.extend(logits.cpu().numpy() > 0.5)
            true_labels.extend(labels.cpu().numpy())
    print()


    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    precision_yes = precision_score(true_labels, predictions, pos_label=1)
    recall_yes = recall_score(true_labels, predictions, pos_label=1)
    precision_no = precision_score(true_labels, predictions, pos_label=0)
    recall_no = recall_score(true_labels, predictions, pos_label=0)

    print('\nCheckpoint 3.1')
    print(f"Overall: acc: {accuracy:.3f}, f1: {f1:.3f}")
    print(f"    Yes: prec: {precision_yes:.3f}, rec: {recall_yes:.3f}, f1: {2*(precision_yes*recall_yes)/(precision_yes+recall_yes):.3f}")
    print(f"     No: prec: {precision_no:.3f}, rec: {recall_no:.3f}, f1: {2*(precision_no*recall_no)/(precision_no+recall_no):.3f}")

    # Regression
    train_dataset = emo_dataset['train']
    validate_dataset = emo_dataset['validation']
    test_dataset = emo_dataset['test']

    regression = Regression(model).to(device)#.to(dtype=torch.float16)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(regression.parameters(), lr=2e-5)

    train_losses = []
    print('Training Regression')
    for epoch in range(5):
        regression.train()
        print(f"\nEpoch {epoch}", end=' ')
        epoch_loss = 0
        for i, batch in enumerate(DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)):
            if i % 100 == 0: print(f"\nBatch {i} - {i+100} ...", end=' ')
            optimizer.zero_grad()
            inputs = tokenizer(batch['text'], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = regression(**inputs).squeeze()

            loss = loss_function(logits, labels)

            # print('inputs', inputs)
            # print('labels', labels)
            # print('logits', logits)
            # print('loss', loss)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_dataset))
        print(f"\nEpoch {epoch} loss: {epoch_loss / len(train_dataset)}")

    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('3.2.png')

    # Evaluation
    regression.eval()
    validate_preds = []
    validate_labels = []
    test_preds = []
    test_labels = []
    print('Evaluating Regression', end=' ')
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(validate_dataset, batch_size=8, pin_memory=True, num_workers=4)):
            if i % 100 == 0: print(f"\nBatch {i} - {i+100} ...", end=' ')
            inputs = tokenizer(batch['text'], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = regression(**inputs).squeeze()

            validate_preds.extend(logits.cpu().numpy())
            validate_labels.extend(labels.cpu().numpy().squeeze())

        for i, batch in enumerate(DataLoader(test_dataset, batch_size=8, pin_memory=True, num_workers=4)):
            if i % 100 == 0: print(f"\nBatch {i} - {i+100} ...", end=' ')
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = regression(**inputs).squeeze()

            test_preds.extend(logits.cpu().numpy())
            test_labels.extend(labels.cpu().numpy().squeeze())
    validate_mae = sum(abs(p - t) for p, t in zip(validate_preds, validate_labels)) / len(validate_preds)
    validate_r = np.corrcoef(validate_preds, validate_labels)[0, 1]

    test_mae = sum(abs(p - t) for p, t in zip(test_preds, test_labels)) / len(test_preds)
    test_r = np.corrcoef(test_preds, test_labels)[0, 1]

    print('\nCheckpoint 3.2')
    print(f"Validation: MAE: {validate_mae:.3f}, R: {validate_r:.3f}")
    print(f"Test: MAE: {test_mae:.3f}, R: {test_r:.3f}")