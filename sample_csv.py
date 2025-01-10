#!/usr/bin/env python3

import argparse
import os
import json
import torch
import torch.amp as amp
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


def train_model(model,
                optimizer,
                scheduler,
                criterion,
                train_dataloader,
                num_epochs,
                device,
                output_size,
                use_profiler):

    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    print(f"Number of parameters: {model.count_parameters()}")

    def _training_loop():
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for i, inputs in enumerate(train_dataloader):
                train_inputs = inputs[:, :, :-1].to(device, non_blocking=True)
                train_labels = inputs[:, :, -1:].to(
                    device,
                    non_blocking=True).squeeze(-1)

                optimizer.zero_grad()
                with amp.autocast(device_type=device.type,
                                  enabled=(device.type == 'cuda')):
                    outputs = model(train_inputs)
                    outputs = outputs.view(-1, output_size)
                    labels = train_labels.view(-1).long()
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                epoch_loss += loss.item()

            print(f"Epoch: {epoch+1}/{num_epochs}, " +
                  f"Loss: {epoch_loss/len(train_dataloader):.4f}")

    if use_profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            with_stack=True
        ) as prof:
            _training_loop(prof)
            print(prof.key_averages().table(
                  sort_by="cuda_time_total,cpu_time_total", row_limit=10))
    else:
        _training_loop()


def main(args):
    # Load Hyperparameters
    with open(args.hyperparameter_file, "r") as f:
        hyperparameters = json.load(f)

    window_size = hyperparameters["window_size"]
    stride = hyperparameters["stride"]
    batch_size = hyperparameters["batch_size"]
    learning_rate = hyperparameters["learning_rate"]
    num_epochs = hyperparameters["num_epochs"]
    d_model = hyperparameters["d_model"]
    d_state = hyperparameters["d_state"]
    conv_kernel = hyperparameters["conv_kernel"]
    expansion_factor = hyperparameters["expansion_factor"]
    num_layers = hyperparameters["num_layers"]
    use_profiler = hyperparameters.get("use_profiler", False)
    lr_min = hyperparameters.get("lr_min", 0.00001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use {device}")

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
                                  num_workers=min(os.cpu_count(), 4),
                                  pin_memory=True,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=min(os.cpu_count(), 4),
                                 pin_memory=True,
                                 shuffle=False)

    input_size = len(train_dataset[0][0]) - 1
    output_size = 2

    # Model instantiation
    model = Model(d_model,
                  d_state,
                  expansion_factor,
                  num_layers,
                  input_size,
                  output_size,
                  conv_kernel).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=num_epochs,
                                                     eta_min=lr_min)

    criterion = nn.CrossEntropyLoss()

    # Training
    train_model(model, optimizer, scheduler, criterion, train_dataloader,
                num_epochs, device, output_size, use_profiler)

    # Test
    model.eval()
    with torch.no_grad(), open(args.output_file, 'w') as f:
        f.write("Test Inputs | Predicted | True Labels\n")
        for inputs in test_dataloader:
            test_inputs = inputs[:, :, :-1].to(device, non_blocking=True)
            test_labels = inputs[:, :, -1:].squeeze(-1).cpu().long()
            logits = model(test_inputs).cpu()
            probabilities = torch.softmax(logits, dim=2)
            predicted = torch.argmax(probabilities, dim=2)

            batch_size = test_inputs.shape[0]

            for batch_idx in range(batch_size):
                for i in range(test_inputs.shape[1]):
                    row_str = f"{test_inputs[batch_idx][i].tolist()} | " + \
                              f"{predicted[batch_idx][i].item()} | " + \
                              f"{test_labels[batch_idx][i].item()}"
                    f.write(row_str + "\n")

    print(f"Test results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test an SSM-NN model.")
    parser.add_argument("train_data",
                        type=str,
                        help="Path to the training data CSV file.")
    parser.add_argument("test_data",
                        type=str,
                        help="Path to the test data CSV file.")
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        default="test_results.txt",
                        help="Path to the output file for test results.")
    parser.add_argument("-p",
                        "--hyperparameter_file",
                        type=str,
                        default="hyperparameters.json",
                        help="Path to the hyperparameter JSON file.")

    args = parser.parse_args()
    main(args)
