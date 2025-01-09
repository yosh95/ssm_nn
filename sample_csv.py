#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ssm_nn.model import Model


class CSVDataset(Dataset):

    def __init__(self,
                 csv_file,
                 window_size,
                 stride,
                 skip_header=False):
        self.csv_file = csv_file
        self.window_size = window_size
        self.stride = stride
        self.skip_header = skip_header
        self.start_indices = self._calculate_start_indices()
        self.data = self._load_data()

    def _calculate_start_indices(self):
        with open(self.csv_file, 'r') as f:
            num_rows = sum(1 for _ in f)
            if self.skip_header:
                num_rows -= 1

        start_indices = []
        for i in range(0, num_rows - self.window_size + 1, self.stride):
            start_indices.append(i)
        return start_indices

    def _load_data(self):
        data = []
        with open(self.csv_file, 'r') as f:
            if self.skip_header:
                next(f)
            for line in f:
                row = [float(x) for x in line.strip().split(',')]
                data.append(row)
        return data

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_data = self.data[start_index:start_index + self.window_size]
        return torch.tensor(window_data, dtype=torch.float32)


def main(args):

    # Hyper parameters
    window_size = 100
    stride = 50
    batch_size = 1

    learning_rate = 0.001
    num_epochs = 100

    d_model = 16
    d_state = 2
    expansion_factor = 2
    num_layers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset instantiation
    train_dataset = CSVDataset(csv_file=args.train_data,
                               window_size=window_size,
                               stride=stride,
                               skip_header=True)

    test_dataset = CSVDataset(csv_file=args.test_data,
                              window_size=window_size,
                              stride=window_size,
                              skip_header=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    input_size = len(train_dataset[0][0]) - 1
    output_size = 2

    # Model instantiation
    model = Model(d_model,
                  d_state,
                  expansion_factor,
                  num_layers,
                  input_size,
                  output_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    # Training
    print(f"Number of parameters: {model.count_parameters()}")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for i, inputs in enumerate(train_dataloader):
            train_inputs = inputs[:, :, :-1].to(device)
            train_labels = inputs[:, :, -1:].to(device).squeeze(-1)

            optimizer.zero_grad()
            outputs = model(train_inputs)
            outputs = outputs.view(-1, output_size)
            labels = train_labels.view(-1).long()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch: {epoch+1}/{num_epochs}, " +
              f"Loss: {epoch_loss/len(train_dataloader):.4f}")

    # Test
    model.eval()
    with torch.no_grad():
        print("Test Inputs | Predicted | True Labels")
        for inputs in test_dataloader:
            test_inputs = inputs[:, :, :-1].to(device)
            test_labels = inputs[:, :, -1:].squeeze(-1)
            logits = model(test_inputs).squeeze().cpu()
            probabilities = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            test_labels = test_labels.long()

            test_inputs_list = test_inputs.cpu().squeeze(0).tolist()
            predicted_list = predicted.tolist()
            test_labels_list = test_labels.squeeze(0).tolist()

            for i in range(len(test_inputs_list)):
                row_str = f"{test_inputs_list[i]} | " + \
                          f"{predicted_list[i]} | {test_labels_list[i]}"
                print(row_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test an SSM-NN model.")
    parser.add_argument("train_data",
                        type=str,
                        help="Path to the training data CSV file.")
    parser.add_argument("test_data",
                        type=str,
                        help="Path to the test data CSV file.")
    args = parser.parse_args()
    main(args)
