import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset into a Pandas DataFrame
# df = pd.read_csv('tweets.csv')

# Split the dataset into a training set and a development set
# train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)

train_data_es = pd.read_csv("translated_train_es_labeled.csv")
dev_data_es = pd.read_csv("translated_dev_es_labeled.csv")

# Load the Roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained('hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021')

# Tokenize the text data
train_encodings = tokenizer(list(train_data_es['translated_text']), truncation=True, padding=True)
dev_encodings = tokenizer(list(dev_data_es['translated_text']), truncation=True, padding=True)

# Map the labels to integers
label_map = {'non-sexist': 0, 'sexist': 1}
train_labels = [label_map[label] for label in train_data_es['label']]
dev_labels = [label_map[label] for label in dev_data_es['label']]


# Convert the dataset into a PyTorch Dataset
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TwitterDataset(train_encodings, train_labels)
dev_dataset = TwitterDataset(dev_encodings, dev_labels)

# Load the RobertaForSequenceClassification model
model = RobertaForSequenceClassification.from_pretrained('hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021', num_labels=2)

# Set up the device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set the number of epochs
num_epochs = 3

# Set the loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Set the best development accuracy to a negative infinity value
best_dev_acc = float('-inf')

# Iterate over the epochs
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    train_loss = 0
    train_acc = 0

    # Iterate over the training set
    for batch in torch.utils.data.DataLoader(train_dataset, batch_size=16):
        # Move the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])

        # Compute the training loss and accuracy
        loss = criterion(outputs.logits.squeeze(), batch['labels'].float())
        train_loss += loss.item()
        train_acc += accuracy_score(batch['labels'].cpu(), torch.round(torch.sigmoid(outputs.logits)).cpu())

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Compute the average training loss and accuracy
    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)

    print(f"Epoch {epoch+1}: Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")

    # Set the model to evaluation mode
    model.eval()

    dev_loss = 0
    dev_acc = 0

    # Turn off gradient calculations for evaluation
    with torch.no_grad():
        # Iterate over the development set
        for batch in torch.utils.data.DataLoader(dev_dataset, batch_size=16):
            # Move the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])

            # Compute the development loss and accuracy
            loss = criterion(outputs.logits.squeeze(), batch['labels'].float())
            dev_loss += loss.item()
            dev_acc += accuracy_score(batch['labels'].cpu(), torch.round(torch.sigmoid(outputs.logits)).cpu())

        # Compute the average development loss and accuracy
        dev_loss /= len(dev_dataset)
        dev_acc /= len(dev_dataset)

        print(f"Epoch {epoch+1}: Development loss: {dev_loss:.4f}, Development accuracy: {dev_acc:.4f}")

        # Save the model if it has the best development accuracy so far
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), 'best_model.pth')