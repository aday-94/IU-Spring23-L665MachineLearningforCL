from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from sklearn.metrics import classification_report

tokenizer = AutoTokenizer.from_pretrained("hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021")

train_data_es = pd.read_csv("translated_train_es_labeled.csv")
dev_data_es = pd.read_csv("translated_dev_es_labeled.csv")


def tokenize_function(examples):
  return tokenizer(examples["translated_text"], padding="max_length", truncation=True, max_length=128)


train_dataset = Dataset.from_pandas(train_data_es)
dev_dataset = Dataset.from_pandas(dev_data_es)

tokenized_train_data = train_dataset.map(tokenize_function, batched=True)
tokenized_dev_data = dev_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("hackathon-pln-es/twitter_sexismo-finetuned-robertuito-exist2021", num_labels=2)

metric = evaluate.load("leslyarun/fbeta_score")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "f1": report["macro avg"]["f1-score"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f2": metric.compute(predictions=predictions, references=labels, beta=2),
    }


training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  num_train_epochs=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_dev_data,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

trainer.save_model("output")