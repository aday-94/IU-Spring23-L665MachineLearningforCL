from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Load the base models and tokenizers
model_es = AutoModelForSequenceClassification.from_pretrained('NLP-LTU/bertweet-large-sexism-detector')
tokenizer_es = AutoTokenizer.from_pretrained('NLP-LTU/bertweet-large-sexism-detector')

model_en = AutoModelForSequenceClassification.from_pretrained('robertou2/twitter_sexismo-finetuned-robertuito-exist2021')
tokenizer_en = AutoTokenizer.from_pretrained('robertou2/twitter_sexismo-finetuned-robertuito-exist2021')

# Define a function to preprocess a tweet and generate predicted probabilities
def predict_proba(tweet, model, tokenizer):
    inputs = tokenizer.encode_plus(tweet, return_tensors='pt', padding=True, truncation=True)
    logits = model(**inputs).logits
    probas = tf.nn.softmax(logits, axis=1).numpy()[0]
    return probas

# Load the labeled dataset for both English and Spanish
train_es = pd.read_csv('translated_train_es_labeled.csv')
train_en = pd.read_csv('translated_train_en_labeled.csv')

dev_es = pd.read_csv('translated_dev_es_labeled.csv')
dev_en = pd.read_csv('translated_dev_en_labeled.csv')

# Generate predicted probabilities for each tweet in the training and development datasets
train_es_probas = [predict_proba(tweet, model_es, tokenizer_es) for tweet in train_es['translated_text']]
train_en_probas = [predict_proba(tweet, model_en, tokenizer_en) for tweet in train_en['translated_text']]

dev_es_probas = [predict_proba(tweet, model_es, tokenizer_es) for tweet in dev_es['translated_text']]
dev_en_probas = [predict_proba(tweet, model_en, tokenizer_en) for tweet in dev_en['translated_text']]

# Concatenate the predicted probabilities from the two models for each tweet
train_probas = tf.concat([train_es_probas, train_en_probas], axis=1)
dev_probas = tf.concat([dev_es_probas, dev_en_probas], axis=1)

# Define the multi-input neural network architecture
inputs = keras.Input(shape=(train_probas.shape[1],))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the concatenated predicted probabilities and ground truth labels
model.fit(train_probas, train_es['label'], validation_data=(dev_probas, dev_es['label']), epochs=10, batch_size=32)

# Evaluate the performance of the model on the development dataset
loss, acc = model.evaluate(dev_probas, dev_es['label'], batch_size=32)
print('Dev loss: {:.4f}, accuracy: {:.4f}'.format(loss, acc))