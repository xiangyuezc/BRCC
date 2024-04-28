import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import time
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = []
        self.labels = []
        self.tokenizer = tokenizer

        with open(data_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().split('\t')
                text = line[0]
                label = int(line[1])
                self.data.append(text)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    predictions = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            predicted_labels = logits.argmax(dim=1)

            predictions.extend(predicted_labels.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

            loss = nn.CrossEntropyLoss()(logits, batch_labels)
            total_loss += loss.item()

            total_correct += (predicted_labels == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, labels, predictions

def calculate_metrics(labels, predictions, num_labels):
    # Calculate evaluation metrics
    confusion = confusion_matrix(labels, predictions)
    accuracy = np.zeros(num_labels)
    precision = np.zeros(num_labels)
    recall = np.zeros(num_labels)
    f1 = np.zeros(num_labels)

    for i in range(num_labels):
        true_indices = np.where(np.array(labels) == i)[0]
        pred_indices = np.where(np.array(predictions) == i)[0]

        if len(true_indices) == 0:
            precision[i] = 0
            recall[i] = 0
        else:
            accuracy[i] = accuracy_score([1 if x == i else 0 for x in labels], [1 if x == i else 0 for x in predictions])
            precision[i] = precision_score([1 if x == i else 0 for x in labels], [1 if x == i else 0 for x in predictions], zero_division=0, average='macro')
            recall[i] = recall_score([1 if x == i else 0 for x in labels], [1 if x == i else 0 for x in predictions], zero_division=0, average='macro')
            f1[i] = f1_score([1 if x == i else 0 for x in labels], [1 if x == i else 0 for x in predictions], zero_division=0, average='macro')

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': confusion
    }

    total_accuracy = np.mean(accuracy)
    total_precision = np.mean(precision)
    total_recall = np.mean(recall)
    total_f1 = np.mean(f1)

    print("Overall Metrics:")
    print(f"Accuracy: {total_accuracy}")
    print(f"Precision: {total_precision}")
    print(f"Recall: {total_recall}")
    print(f"F1 Score: {total_f1}")
    print()

    for i in range(num_labels):
        label_accuracy = confusion[i, i] / np.sum(confusion[i])
        label_precision = precision[i]
        label_recall = recall[i]
        label_f1 = f1[i]

        metrics[i] = {
            'Accuracy': label_accuracy,
            'Precision': label_precision,
            'Recall': label_recall,
            'F1 Score': label_f1
        }

    return metrics

def predict_model(test_file, model_path):
    # Load the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and create the test dataset and data loader
    tokenizer = BertTokenizer.from_pretrained('E:\\PythonProgram\\Merge-Bert-Chinese-Text-Classification\\bert-base-chinese')
    test_dataset = CustomDataset(test_file, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Create the model
    num_labels = len(set(test_dataset.labels))
    model = BertForSequenceClassification.from_pretrained('E:\\PythonProgram\\Merge-Bert-Chinese-Text-Classification\\bert-base-chinese', num_labels=num_labels)

    # Load the pre-trained model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model.")
    else:
        print("Model not found. Please run training first.")

    model = model.to(device)
    model.eval()

    predictions = []
    labels = []
    texts = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].squeeze().to(device)
            attention_mask = batch['attention_mask'].squeeze().to(device)
            label = batch['label'].item()
            text = test_dataset.data[i]

            outputs = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0)
            )

            logits = outputs.logits
            predicted_label = logits.argmax().item()

            predictions.append(predicted_label)
            labels.append(label)
            texts.append(text)

    # Set print options
    np.set_printoptions(linewidth=np.inf)

    metrics = calculate_metrics(labels, predictions, num_labels)

    print('Metrics_LABEL')
    print('Accuracy', metrics['Accuracy'])
    print('Precision', metrics['Precision'])
    print('Recall', metrics['Recall'])
    print('F1 Score', metrics['F1 Score'])

    print("Original Labels:", labels)
    print("Predicted Labels:", predictions)

    print("Results (Incorrectly Predicted):")
    print("-------------------------------------------------")
    print("| Index | Original Label | Predicted Label | Text |  ")
    print("-------------------------------------------------")
    num = 0
    for i in range(len(labels)):
        if labels[i] != predictions[i]:
            num += 1
            print(f"|{num:<5}|{i+1:<5}|{labels[i]:<2}|{predictions[i]:<2}|| {texts[i]:<10}|")
    print("-------------------------------------------------")


test_file = "test.txt"
model_path = "BERTOnly_model.pth"
epochs = 10
batch_size = 32
dropout = 0.2
learning_rate = 2e-5


predict_model(test_file, model_path)
