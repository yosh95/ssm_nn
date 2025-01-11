#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from ssm_nn.bpe import BPE
from ssm_nn.model import Model


with open("text_data.txt", "r") as f:
    text = f.read().strip()

words = Counter(text.split())

vocab_size = 3000
bpe = BPE(vocab_size=vocab_size)
bpe.learn_merges(words)

tokens = bpe.tokenize(text)


class TextDataset(Dataset):
    def __init__(self, tokens, vocab, seq_len):
        self.tokens = tokens
        self.vocab = list(vocab)
        self.vocab_to_id = {token: id for id, token in enumerate(self.vocab)}
        self.ids = [self.vocab_to_id.get(token, 0) for token in self.tokens]
        self.seq_len = seq_len

        self.sequences = []
        for i in range(0, len(self.ids) - seq_len, 1):
            self.sequences.append(self.ids[i: i + seq_len + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1],
                            dtype=torch.long), torch.tensor(sequence[1:],
                                                            dtype=torch.long)


seq_len = 64
dataset = TextDataset(tokens, bpe.vocab, seq_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

d_model = 32
d_state = 4
expansion_factor = 2
num_layers = 2
input_size = len(bpe.vocab)
output_size = len(bpe.vocab)
conv_kernel = 3

model = Model(d_model,
              d_state,
              expansion_factor,
              num_layers,
              input_size,
              output_size,
              conv_kernel)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(
            nn.functional.one_hot(inputs,
                                  num_classes=len(bpe.vocab)).float())

        loss = criterion(outputs.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch: {epoch + 1}/{epochs}, " +
          f"Loss: {total_loss / len(dataloader):.4f}")


def generate_text(model,
                  start_text,
                  bpe,
                  device,
                  max_length=100,
                  temperature=1.0):
    model.eval()
    start_tokens = bpe.tokenize(start_text)
    start_ids = [bpe.vocab_to_id[token] for token in start_tokens]
    generated_ids = start_ids[:]

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor(generated_ids).unsqueeze(0).to(device)
            output = model(
                nn.functional.one_hot(
                        input_tensor,
                        num_classes=len(bpe.vocab)).float())
            output = output[:, -1, :] / temperature
            probs = F.softmax(output, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_id)
            if next_id == bpe.vocab_to_id["<end>"]:
                break

    generated_tokens = [bpe.id_to_vocab[idx] for idx in generated_ids]
    generated_text = " ".join(generated_tokens)
    return generated_text


start_text = "You can"
generated_text = generate_text(model,
                               start_text,
                               bpe,
                               device,
                               100)
print(f"\nGenerated Text: {generated_text}")
