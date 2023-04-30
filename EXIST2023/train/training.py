import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Step 1: Load the training and development datasets for both English and Spanish
train_data_en = pd.read_csv("translated_train_en.csv")
train_data_es = pd.read_csv("translated_train_es.csv")
dev_data_en = pd.read_csv("translated_dev_en.csv")
dev_data_es = pd.read_csv("translated_dev_es.csv")

# Step 2: Load the pre-trained models for English and Spanish
tokenizer_en = AutoTokenizer.from_pretrained("NLP-LTU/bertweet-large-sexism-detector")
model_en = AutoModelForSequenceClassification.from_pretrained(
    "NLP-LTU/bertweet-large-sexism-detector")
tokenizer_es = AutoTokenizer.from_pretrained("hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021")
model_es = AutoModelForSequenceClassification.from_pretrained("hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021")


# Step 3: Define a custom neural network model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(768 * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input_en, input_es):
        output_en = model_en(input_en)[0]
        output_es = model_es(input_es)[0]
        combined_output = torch.cat([output_en.mean(dim=1), output_es.mean(dim=1)], dim=1)
        x = nn.functional.relu(self.fc1(combined_output))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# Step 4: Define a loss function and an optimizer for your custom model
model = MyModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Loop through the training data and train your custom model
for epoch in range(10):
    for i, (en_tweet, es_tweet, label) in enumerate(
            zip(train_data_en["translated_text"], train_data_es["translated_text"], train_data_en["label"])):
        # Preprocess the tweets for English and Spanish
        input_en = tokenizer_en.encode_plus(en_tweet, add_special_tokens=True, return_tensors="pt")
        input_es = tokenizer_es.encode_plus(es_tweet, add_special_tokens=True, return_tensors="pt")

        # Generate the outputs from the pre-trained models for English and Spanish
        with torch.no_grad():
            output_en = model_en(input_en["input_ids"], input_en["attention_mask"])[0]
            output_es = model_es(input_es["input_ids"], input_es["attention_mask"])[0]

        # Pass the outputs through the custom model and update the parameters
        model.zero_grad()
        outputs = model(output_en, output_es)
        label = torch.tensor([label], dtype=torch.float32)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

# Step 6: Evaluate the performance of your model on the development data
with torch.no_grad():
    num_correct = 0
    num_total = 0
    for i, (en_tweet, es_tweet, label) in enumerate(zip(dev_data_en["translated_text"], dev_data_es["translated_text"], dev_data_en["label"])):
        # Preprocess the tweets for English and Spanish
        input_en = tokenizer_en.encode_plus(en_tweet, add_special_tokens=True, return_tensors="pt")
        input_es = tokenizer_es.encode_plus(es_tweet, add_special_tokens=True, return_tensors="pt")

        # Generate the outputs from the pre-trained models for English and Spanish
        output_en = model_en(input_en["input_ids"], input_en["attention_mask"])[0]
        output_es = model_es(input_es["input_ids"], input_es["attention_mask"])[0]

        # Pass the outputs through the custom model and compute the predictions
        outputs = model(output_en, output_es)
        predicted_label = int(outputs[0] > 0.5)

        # Update the counters for correct and total predictions
        num_correct += int(predicted_label == label)
        num_total += 1

# Compute the accuracy of the custom model on the development data
accuracy = num_correct / num_total
print("Accuracy on development data: {:.2f}%".format(accuracy * 100))



