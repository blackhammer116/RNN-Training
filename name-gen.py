import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import matplotlib.pyplot as plt


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

class VRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
      super().__init__()
      self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
      self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x):
      out, _ = self.rnn(x)
      out = self.lin(out)
      return out

def preprocess_data(data, seq_length):
    chars = sorted(list(set(data)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    # idx2char = {i: ch for ch, i in char2idx.items()}
    # vocab_size = len(chars)

    seq_length = 4
    x_data, y_data = [], []

    for i in range(len(data) - seq_length):
        x_seq = data[i:i+seq_length]
        y_seq = data[i+1:i+seq_length+1]
        x_data.append([char2idx[c] for c in x_seq])
        y_data.append([char2idx[c] for c in y_seq])

    x_train = torch.tensor(x_data)
    y_train = torch.tensor(y_data)

    return x_train, y_train

def one_hot(x, num_classes):
    return torch.eye(num_classes)[x]


def train(model, x_train, y_train, vocab_size, epochs=1000, lr=0.001):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        inputs = one_hot(x_train, vocab_size).float().to(device)
        labels = y_train.to(device)
        outputs = model(inputs)

        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


def generate_name(model, prefix, char2idx, idx2char, seq_length):
    generated_name = prefix
    input_seq = [char2idx[c] for c in prefix]

    model.eval()
    with torch.no_grad():
        while len(generated_name) < 20:
            # Take the last seq_length characters
            if len(input_seq) < seq_length:
                # Pad with a special character or handle shorter prefixes
                current_input = input_seq
            else:
                current_input = input_seq[-seq_length:]


            input_tensor = one_hot(torch.tensor(current_input).unsqueeze(0), len(char2idx)).float().to(device)
            output = model(input_tensor)

            next_char_probs = output[:, -1, :]
            next_char_index = torch.multinomial(torch.softmax(next_char_probs, dim=-1), 1).item()
            next_char = idx2char[next_char_index]

            generated_name += next_char
            input_seq.append(next_char_index)

            if next_char == '\n':
                break

    return generated_name.strip()


with open("names.txt", "r") as f:
    data = f.read()

chars = sorted(list(set(data)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
seq_length = 3
x_train, y_train = preprocess_data(data, seq_length)
vocab_size = len(chars)

input_size = vocab_size
hidden_size = 128
output_size = vocab_size

model = VRNN(input_size, hidden_size, output_size).to(device)
print(model)

train(model, x_train, y_train, vocab_size)


prefixes = ["Al", "Bo", "Ch", "Za", "Ji", "Lu", "Re"]
for prefix in prefixes:
    generated_name = generate_name(model, prefix, char2idx, idx2char, seq_length)
    print(f"Generated name with prefix '{prefix}': {generated_name}")
