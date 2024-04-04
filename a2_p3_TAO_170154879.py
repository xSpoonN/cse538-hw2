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

def fineTune_boolQ(model, name = 'model'):
    train_dataset = boolq_dataset['train'].map(lambda example: {'label': 1 if example['answer'] == True else 0})
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_losses = []
    print(f'Fine-tuning {name} on BoolQ')
    for epoch in range(5):
        model.train()
        print(f"\nEpoch {epoch}", end=' ')
        epoch_loss = 0
        for i, batch in enumerate(DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)):
            if i % 200 == 0: print(f"\nBatch {i} - {i+200} ...", end=' ')
            optimizer.zero_grad()
            prompts = [f"{passage}\n{question}?\n" for passage, question in zip(batch['passage'], batch['question'])]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)#.to(dtype=torch.float16)
            logits = model(**inputs).squeeze()#.to(dtype=torch.float16)

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_dataset))
        print(f"\nEpoch {epoch} loss: {epoch_loss / len(train_dataset)}")
    return train_losses

def evaluate_boolQ(model, name = 'model'):
    validate_dataset = boolq_dataset['validation'].map(lambda example: {'label': 1 if example['answer'] == True else 0})
    model.eval()
    predictions = []
    true_labels = []
    print(f'Evaluating {name} on BoolQ', end=' ')
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(validate_dataset, batch_size=16, pin_memory=True, num_workers=4)):
            if i % 200 == 0: print(f"\nBatch {i} - {i+200} ...", end=' ')
            prompts = [f"{passage}\n{question}?\n" for passage, question in zip(batch['passage'], batch['question'])]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = model(**inputs).squeeze()

            predictions.extend(logits.cpu().numpy() > 0.5)
            true_labels.extend(labels.cpu().numpy())
        print()

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        precision_yes = precision_score(true_labels, predictions, pos_label=1)
        recall_yes = recall_score(true_labels, predictions, pos_label=1)
        precision_no = precision_score(true_labels, predictions, pos_label=0)
        recall_no = recall_score(true_labels, predictions, pos_label=0)
    return accuracy, f1, precision_yes, recall_yes, precision_no, recall_no

def fineTune_emoBank(model, name = 'model'):
    train_dataset = emo_dataset['train']
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_losses = []
    print(f'Fine-tuning {name} on EmoBank')
    for epoch in range(5):
        model.train()
        print(f"\nEpoch {epoch}", end=' ')
        epoch_loss = 0
        for i, batch in enumerate(DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)):
            if i % 200 == 0: print(f"\nBatch {i} - {i+200} ...", end=' ')
            optimizer.zero_grad()
            inputs = tokenizer(batch['text'], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = model(**inputs).squeeze()

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_dataset))
        print(f"\nEpoch {epoch} loss: {epoch_loss / len(train_dataset)}")
    return train_losses

def evaluate_emoBank(model, name = 'model'):
    validate_dataset = emo_dataset['validation']
    test_dataset = emo_dataset['test']
    model.eval()
    validate_preds = []
    validate_labels = []
    test_preds = []
    test_labels = []
    print(f'Evaluating {name} on EmoBank', end=' ')
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(validate_dataset, batch_size=8, pin_memory=True, num_workers=4)):
            if i % 200 == 0: print(f"\nBatch {i} - {i+200} ...", end=' ')
            inputs = tokenizer(batch['text'], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = model(**inputs).squeeze()

            validate_preds.extend(logits.cpu().numpy())
            validate_labels.extend(labels.cpu().numpy().squeeze())

        for i, batch in enumerate(DataLoader(test_dataset, batch_size=8, pin_memory=True, num_workers=4)):
            if i % 200 == 0: print(f"\nBatch {i} - {i+200} ...", end=' ')
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt').to(device)
            labels = batch['label'].float().to(device)
            logits = model(**inputs).squeeze()

            test_preds.extend(logits.cpu().numpy())
            test_labels.extend(labels.cpu().numpy().squeeze())
    validate_mae = sum(abs(p - t) for p, t in zip(validate_preds, validate_labels)) / len(validate_preds)
    validate_r = np.corrcoef(validate_preds, validate_labels)[0, 1]

    test_mae = sum(abs(p - t) for p, t in zip(test_preds, test_labels)) / len(test_preds)
    test_r = np.corrcoef(test_preds, test_labels)[0, 1]
    return validate_mae, validate_r, test_mae, test_r

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    boolq_dataset = load_dataset('google/boolq')
    emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence', trust_remote_code=True)
    model = RobertaModel.from_pretrained('distilroberta-base').to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base', pad_token='<pad>', unk_token='<unk>')

    if False: # Classification
        classifier = Classifier(model).to(device)#.to(dtype=torch.float16)
        train_losses = fineTune_boolQ(classifier)

        plt.plot(train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('3.1.png')

        # Evaluation
        accuracy, f1, precision_yes, recall_yes, precision_no, recall_no = evaluate_boolQ(classifier)
        print('\nCheckpoint 3.1')
        print(f"Overall: acc: {accuracy:.3f}, f1: {f1:.3f}")
        print(f"    Yes: prec: {precision_yes:.3f}, rec: {recall_yes:.3f}, f1: {2*(precision_yes*recall_yes)/(precision_yes+recall_yes):.3f}")
        print(f"     No: prec: {precision_no:.3f}, rec: {recall_no:.3f}, f1: {2*(precision_no*recall_no)/(precision_no+recall_no):.3f}")

    if False: # Regression
        validate_dataset = emo_dataset['validation']
        test_dataset = emo_dataset['test']

        regression = Regression(model).to(device)#.to(dtype=torch.float16)
        train_losses = fineTune_emoBank(regression)

        plt.plot(train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('3.2.png')

        # Evaluation
        validate_mae, validate_r, test_mae, test_r = evaluate_emoBank(regression)
        print('\nCheckpoint 3.2')
        print(f"Validation: MAE: {validate_mae:.3f}, R: {validate_r:.3f}")
        print(f"Test: MAE: {test_mae:.3f}, R: {test_r:.3f}")

    if True: # Modifications
        distilRB_rand1 = RobertaModel.from_pretrained('distilroberta-base').to(device)
        distilRB_rand2 = RobertaModel.from_pretrained('distilroberta-base').to(device)
        distilRB_KQ1 = RobertaModel.from_pretrained('distilroberta-base').to(device)
        distilRB_KQ2 = RobertaModel.from_pretrained('distilroberta-base').to(device)
        distilRB_nores1 = RobertaModel.from_pretrained('distilroberta-base').to(device)
        distilRB_nores2 = RobertaModel.from_pretrained('distilroberta-base').to(device)


        # Modify distilRB-rand
        distilRB_rand1.init_weights()
        distilRB_rand2.init_weights()

        # Modify distilRB-KQ
        for layer in distilRB_KQ1.encoder.layer[-2:]:
            qW = layer.attention.self.query.weight
            kW = layer.attention.self.key.weight
            sharedW = (qW + kW) / 2

            layer.attention.self.query.weight = torch.nn.Parameter(sharedW)
            layer.attention.self.key.weight = torch.nn.Parameter(sharedW)
        for layer in distilRB_KQ2.encoder.layer[-2:]:
            qW = layer.attention.self.query.weight
            kW = layer.attention.self.key.weight
            sharedW = (qW + kW) / 2

            layer.attention.self.query.weight = torch.nn.Parameter(sharedW)
            layer.attention.self.key.weight = torch.nn.Parameter(sharedW)


        # Modify distilRB-nores
        for layer in distilRB_nores1.encoder.layer:
            layer.attention.output = torch.nn.Identity()
            layer.output = layer.attention.output
        for layer in distilRB_nores2.encoder.layer:
            layer.attention.output = torch.nn.Identity()
            layer.output = layer.attention.output

        distilRB_rand1 = Classifier(distilRB_rand1).to(device)
        distilRB_rand2 = Regression(distilRB_rand2).to(device)
        distilRB_KQ1 = Classifier(distilRB_KQ1).to(device)
        distilRB_KQ2 = Regression(distilRB_KQ2).to(device)
        distilRB_nores1 = Classifier(distilRB_nores1).to(device)
        distilRB_nores2 = Regression(distilRB_nores2).to(device)

        # train_losses1 = fineTune_boolQ(distilRB_rand1, 'distilRB_rand')
        # train_losses2 = fineTune_boolQ(distilRB_KQ1, 'distilRB_KQ')
        train_losses3 = fineTune_boolQ(distilRB_nores1, 'distilRB_nores')
        train_losses4 = fineTune_emoBank(distilRB_rand2, 'distilRB_rand')
        train_losses5 = fineTune_emoBank(distilRB_KQ2, 'distilRB_KQ')
        train_losses6 = fineTune_emoBank(distilRB_nores2, 'distilRB_nores')

        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        axs = axs.flatten()
        for i, (ax, train_losses, title) in enumerate(zip(axs, [train_losses1, train_losses2, train_losses3, train_losses4, train_losses5, train_losses6], 
                                                   ['distilRB_rand BoolQ', 'distilRB_KQ BoolQ', 'distilRB_nores BoolQ', 'distilRB_rand EmoBank', 'distilRB_KQ EmoBank', 'distilRB_nores EmoBank'])):
            ax.plot(train_losses)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Loss {title}')
        plt.savefig('3.3.png')