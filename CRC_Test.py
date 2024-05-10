import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time

# Set parameters
num_classes = 7  # Number of label classes
other_features_dim = 3  # Dimension of other features
batch_size = 16  # Batch size
max_length = 200  # Maximum text length

# Create model
class BERTClassifier(nn.Module):
    def __init__(self, num_classes, other_features_dim):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('E:\\PythonProgram\\Merge-Bert-Chinese-Text-Classification\\bert-base-chinese')
        self.classifier = nn.Linear(self.bert.config.hidden_size + other_features_dim, num_classes)

    def forward(self, input_ids, attention_mask, other_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined_features = torch.cat((pooled_output, other_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

# Data loader
class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer, other_features_dim=2, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.other_features_dim = other_features_dim
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace('%','#').replace(' ','').replace('/','').replace(':','').replace('=','等于').replace('\t','\t')
                line = line.strip().split('\t')
                text = line[0]
                other_features = list(map(float, line[1:-1]))
                label = int(line[-1])
                self.data.append((text, other_features, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx][0]
        other_features = self.data[idx][1]
        label = self.data[idx][2]

        encoded_inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded_inputs['input_ids'].squeeze(0)
        attention_mask = encoded_inputs['attention_mask'].squeeze(0)

        other_features += [0.0] * (self.other_features_dim - len(other_features))
        other_features = torch.tensor(other_features)
        label = torch.tensor(label)

        return input_ids, attention_mask, other_features, label

# Custom collate_fn function
def collate_fn(data):
    input_ids_list, attention_mask_list, other_features_list, labels = zip(*data)

    input_ids_padded = torch.stack(input_ids_list)
    attention_mask_padded = torch.stack(attention_mask_list)
    other_features_padded = torch.stack(other_features_list)

    labels_tensor = torch.tensor(labels)

    return input_ids_padded, attention_mask_padded, other_features_padded, labels_tensor

def compute_metrics(y_true, y_pred, num_classes):
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}

    for i in range(num_classes):
        accuracy[i] = accuracy_score([1 if x == i else 0 for x in y_true], [1 if x == i else 0 for x in y_pred])
        precision[i] = precision_score([1 if x == i else 0 for x in y_true], [1 if x == i else 0 for x in y_pred], zero_division=0, average='macro')
        recall[i] = recall_score([1 if x == i else 0 for x in y_true], [1 if x == i else 0 for x in y_pred], zero_division=0, average='macro')
        f1[i] = f1_score([1 if x == i else 0 for x in y_true], [1 if x == i else 0 for x in y_pred], zero_division=0, average='macro')

    return accuracy, precision, recall, f1

def main():
    # Load BERT model and tokenizer
    model = BERTClassifier(num_classes, other_features_dim)
    tokenizer = BertTokenizer.from_pretrained('E:\\PythonProgram\\Merge-Bert-Chinese-Text-Classification\\bert-base-chinese')

    # Load test data
    test_dataset = MyDataset('test_data1.txt', tokenizer, other_features_dim, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Move model to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load pre-trained model
    if os.path.exists('CRC_model.pth'):
        print('Pre-trained model detected, loading BRCC_model.pth for prediction')
        model.load_state_dict(torch.load('CRC_model.pth'))
        model.eval()

        # Evaluate on test set
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, other_features, labels in test_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                other_features = other_features.to(device)
                labels = labels.to(device)

                # Forward pass
                logits = model(input_ids, attention_mask, other_features)

                # Record predictions and labels
                _, predicted = torch.max(logits, 1)
                test_preds.extend(predicted.tolist())
                test_labels.extend(labels.tolist())

        # Print actual labels and predicted labels
        print("Actual Labels:", test_labels)
        print("Predicted Labels:", test_preds)

        # Compute test metrics for each label
        accuracy, precision, recall, f1 = compute_metrics(test_labels, test_preds, num_classes)
        for label in range(num_classes):
            print(f"Label {label}")
            print("Test Accuracy:", accuracy[label])
            print("Test Precision:", precision[label])
            print("Test Recall:", recall[label])
            print("Test F1:", f1[label])
            print()

if __name__ == '__main__':
    main()
