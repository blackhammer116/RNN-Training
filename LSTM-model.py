import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = 'aclImdb'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

def load_reviews_and_labels(directory):
    reviews = []
    labels = []
    for sentiment in ['pos', 'neg']:
        sentiment_dir = os.path.join(directory, sentiment)
        for filename in os.listdir(sentiment_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(sentiment_dir, filename), 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(1 if sentiment == 'pos' else 0)
    return reviews, labels

train_reviews, train_labels = load_reviews_and_labels(train_dir)
test_reviews, test_labels = load_reviews_and_labels(test_dir)

# Initialize a Tokenizer object
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

tokenizer.fit_on_texts(train_reviews)

train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

print("Training sequences created.")
print("Test sequences created.")


# Pad sequences to a fixed length
max_length = 256
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Convert sequences and labels to PyTorch tensors
train_padded = torch.tensor(train_padded, dtype=torch.long)
test_padded = torch.tensor(test_padded, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.float)
test_labels = torch.tensor(test_labels, dtype=torch.float)


print("Training data padded and converted to tensors:", train_padded.shape, train_labels.shape)
print("Test data padded and converted to tensors:", test_padded.shape, test_labels.shape)


# LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :] # Get the output from the last time step
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        return output


embedding_dim = 100
hidden_dim = 256
n_layers = 2

if 'tokenizer' in locals() and hasattr(tokenizer, 'word_index'):
    vocab_size = len(tokenizer.word_index) + 1 # Add 1 for padding or OOV token
    print(f"Using vocab_size: {vocab_size}")
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, n_layers)
    print(model)
else:
    print("Tokenizer or its word_index not found. Cannot initialize model example.")


train_dataset = TensorDataset(train_padded, train_labels)
test_dataset = TensorDataset(test_padded, test_labels)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Number of batches in training dataloader:", len(train_dataloader))
print("Number of batches in test dataloader:", len(test_dataloader))

# Training the model
def train(model, train_dataloader, epochs=5, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to: {device}")
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:  
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

    print("Training finished.")

# Evaluation
def evaluate(model, test_dataloader):
    model.eval()

    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = torch.round(outputs.squeeze())
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Accuracy on the test set: {accuracy:.4f}")
    print(f"F1-score on the test set: {f1:.4f}")

# Making predictions
def make_predictions(model, sample_reviews):

    sample_sequences = tokenizer.texts_to_sequences(sample_reviews)
    sample_padded = pad_sequences(sample_sequences, maxlen=max_length, padding='post', truncating='post')
    sample_tensor = torch.tensor(sample_padded, dtype=torch.long)
    sample_tensor = sample_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(sample_tensor)
        predicted_sentiments = torch.round(outputs.squeeze())

    sentiment_map = {0: "Negative", 1: "Positive"}

    print("Sentiment predictions for sample reviews:")
    for i, review in enumerate(sample_reviews):
        predicted_label = sentiment_map[predicted_sentiments[i].item()]
        print(f"Review: '{review[:80]}...'")
        print(f"Predicted Sentiment: {predicted_label}\n")


print("Starting training...")
train(model, train_dataloader, epochs=5, lr=0.001)
print("Training completed.\n\n")
evaluate(model, test_dataloader)

sample_reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "The acting was terrible and the plot was boring. A complete waste of time.",
    "It was an okay film, nothing special but not bad either.",
    "The visual effects were stunning, but the story was weak.",
    "Best movie I've seen in years. Highly recommended!"
]
print("\nMaking predictions on sample reviews...\n")
make_predictions(model, sample_reviews)
