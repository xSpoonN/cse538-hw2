import sys
import torch
from torch.utils.data import DataLoader
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
        sequence_output = output.last_hidden_state
        pooled_output = sequence_output[:, 0]
        pooled_output = output.pooler_output
        logits = self.linear(pooled_output)
        return logits
    
class Regression(torch.nn.Module):
    def __init__(self, basemodel):
        super().__init__()
        self.model = basemodel
        self.linear = torch.nn.Linear(basemodel.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        logits = self.linear(pooled_output)
        return logits

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    boolq_dataset = load_dataset('google/boolq')
    emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence', trust_remote_code=True)
    model = RobertaModel.from_pretrained('distilroberta-base').to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')

    train_dataset = boolq_dataset['train'].map(lambda example: {'label': 1 if example['answer'] == True else 0})
    validate_dataset = boolq_dataset['validation'].map(lambda example: {'label': 1 if example['answer'] == True else 0})

    classifier = Classifier(model).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)

    train_losses = []
    print('Training Classifier')
    for epoch in range(3):
        classifier.train()
        print(f"\nEpoch {epoch}", end=' ')
        epoch_loss = 0
        for i, batch in enumerate(DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)):
            if i % 100 == 0: print(f"\nBatch {i} - {i+100}", end=' ')
            print('.', end='', flush=True)
            optimizer.zero_grad()
            inputs = tokenizer(batch['passage'], batch['question'], padding=True, truncation=True, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = classifier(**inputs).squeeze()

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
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
            if i % 100 == 0: print(f"\nBatch {i} - {i+100}", end=' ')
            print('.', end='', flush=True)
            optimizer.zero_grad()
            inputs = tokenizer(batch['passage'], batch['question'], padding=True, truncation=True, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = classifier(**inputs).squeeze()

            preds = torch.round(torch.sigmoid(logits)).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())


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

    regression = Regression(model).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(regression.parameters(), lr=1e-4)

    train_losses = []
    for epoch in range(3):
        regression.train()
        print(f"\nEpoch {epoch}", end=' ')
        epoch_loss = 0
        for i, batch in enumerate(DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)):
            if i % 100 == 0: print(f"\nBatch {i} - {i+100}", end=' ')
            print('.', end='', flush=True)
            optimizer.zero_grad()
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').to(device)
            labels = batch['label'].unsqueeze(1).to(device)
            logits = regression(**inputs).squeeze(-1)

            loss = loss_function(logits, labels)
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
        for i, batch in enumerate(DataLoader(validate_dataset, batch_size=16, pin_memory=True, num_workers=4)):
            if i % 100 == 0: print(f"\nBatch {i} - {i+100}", end=' ')
            print('.', end='', flush=True)
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').to(device)
            labels = batch['label'].unsqueeze(1).to(device)
            logits = regression(**inputs).squeeze(-1)

            validate_preds.extend(logits.cpu().numpy())
            validate_labels.extend(labels.cpu().numpy().squeeze())

        for i, batch in enumerate(DataLoader(test_dataset, batch_size=16, pin_memory=True, num_workers=4)):
            if i % 100 == 0: print(f"\nBatch {i} - {i+100}", end=' ')
            print('.', end='', flush=True)
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').to(device)
            labels = batch['label'].unsqueeze(1).to(device)
            logits = regression(**inputs).squeeze(-1)

            test_preds.extend(logits.cpu().numpy())
            test_labels.extend(labels.cpu().numpy().squeeze())
    validate_mae = sum(abs(p - t) for p, t in zip(predictions, true_labels)) / len(predictions)
    validate_r = sum(p * t for p, t in zip(predictions, true_labels)) / (sum(p ** 2 for p in predictions) ** 0.5 * sum(t ** 2 for t in true_labels) ** 0.5)

    test_mae = sum(abs(p - t) for p, t in zip(predictions, true_labels)) / len(predictions)
    test_r = sum(p * t for p, t in zip(predictions, true_labels)) / (sum(p ** 2 for p in predictions) ** 0.5 * sum(t ** 2 for t in true_labels) ** 0.5)

    print('\nCheckpoint 3.2')
    print(f"Validation: MAE: {validate_mae:.3f}, R: {validate_r:.3f}")
    print(f"Test: MAE: {test_mae:.3f}, R: {test_r:.3f}")