#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
import torch
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ssm_nn.model import Model
import numpy as np
from sklearn.metrics import roc_curve, auc


class CSVDataset(Dataset):

    def __init__(self,
                 csv_file,
                 window_size,
                 stride,
                 is_train=True):
        self.csv_file = csv_file
        self.window_size = window_size
        self.stride = stride
        self.data = self._load_data(is_train)
        self.start_indices = self._calculate_start_indices()

    def _load_data(self, is_train):
        df = pd.read_csv(self.csv_file)
        if not is_train:
            self.labels = df.iloc[:, -1].astype(float).values

        return df.iloc[:, :-1].astype(float).values

    def _calculate_start_indices(self):
        num_rows = len(self.data)
        if num_rows < self.window_size:
            self.window_size = num_rows
        start_indices = []
        for i in range(0, num_rows - self.window_size + 1, self.stride):
            start_indices.append(i)
        return start_indices

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_data = self.data[start_index:start_index + self.window_size]
        if hasattr(self, 'labels'):
            window_labels = self.labels[start_index:start_index +
                                        self.window_size]
            return torch.tensor(window_data, dtype=torch.float32), \
                torch.tensor(window_labels, dtype=torch.float32)
        else:
            return torch.tensor(window_data, dtype=torch.float32)


def train_model(model,
                optimizer,
                scheduler,
                criterion,
                train_dataloader,
                num_epochs,
                device,
                use_profiler):

    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    print(f"Number of parameters: {model.count_parameters()}")

    def _training_loop():
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for i, inputs in enumerate(train_dataloader):
                train_inputs = inputs.to(device, non_blocking=True)

                optimizer.zero_grad()
                with amp.autocast(device_type=device.type,
                                  enabled=(device.type == 'cuda')):
                    outputs = model(train_inputs)
                    loss = criterion(outputs, train_inputs)

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


def test_model(model, test_dataloader, device, result_file):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad(), open(result_file, 'w') as f:
        f.write("Test Inputs | Predicted | True Labels | Score\n")
        for inputs, labels in test_dataloader:
            test_inputs = inputs.to(device, non_blocking=True)
            test_labels = labels.cpu().long()
            reconstructions = model(test_inputs).cpu()

            loss_fn = nn.MSELoss(reduction='none')
            scores = loss_fn(reconstructions, inputs.cpu()).mean(dim=[1, 2])

            batch_size = test_inputs.shape[0]
            for batch_idx in range(batch_size):
                for i in range(test_inputs.shape[1]):
                    row_str = \
                        f"{test_inputs[batch_idx][i].tolist()} | " + \
                        f"{reconstructions[batch_idx][i].tolist()} | " + \
                        f"{test_labels[batch_idx][i].item()} | " + \
                        f"{scores[batch_idx].item()}"
                    f.write(row_str + "\n")
                    all_scores.append(scores[batch_idx].item())
                    all_labels.append(test_labels[batch_idx][i].item())

    print(f"Test results saved to: {result_file}")
    return np.array(all_scores), np.array(all_labels)


def evaluate_anomaly_detection(all_scores, all_labels):
    # Calculate ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    print(f"AUC Score: {roc_auc:.4f}")


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
                               is_train=True)

    test_dataset = CSVDataset(csv_file=args.test_data,
                              window_size=window_size,
                              stride=window_size,
                              is_train=False)

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
    input_size = len(train_dataset[0][0])
    output_size = input_size

    # Model instantiation
    model = Model(d_model,
                  d_state,
                  expansion_factor,
                  num_layers,
                  input_size,
                  output_size,
                  conv_kernel).to(device)

    if args.model_path:
        print(f"Loading model from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path,
                                         weights_only=True,
                                         map_location=device))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=num_epochs,
                                                         eta_min=lr_min)
        criterion = nn.MSELoss()

        # Training
        train_model(model, optimizer, scheduler, criterion, train_dataloader,
                    num_epochs, device, use_profiler)

        # Save Model
        if args.output_model_path:
            print(f"Saving model to: {args.output_model_path}")
            torch.save(model.state_dict(), args.output_model_path)

    # Test
    all_scores, all_labels = test_model(model,
                                        test_dataloader,
                                        device,
                                        args.result_file)

    # Evaluate
    evaluate_anomaly_detection(all_scores, all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test an SSM-NN model for anomaly detection.")
    parser.add_argument("-t",
                        "--train_data",
                        type=str,
                        default="train_data.csv",
                        help="Path to the training data CSV file.")
    parser.add_argument("-e",
                        "--test_data",
                        type=str,
                        default="test_data.csv",
                        help="Path to the test data CSV file.")
    parser.add_argument("-r",
                        "--result_file",
                        type=str,
                        default="test_results.txt",
                        help="Path to the output file for test results.")
    parser.add_argument("-p",
                        "--hyperparameter_file",
                        type=str,
                        default="hyperparameters.json",
                        help="Path to the hyperparameter JSON file.")
    parser.add_argument("-o",
                        "--output_model_path",
                        type=str,
                        default="model.pth",
                        help="Path to save the trained model.")
    parser.add_argument("-m",
                        "--model_path",
                        type=str,
                        default=None,
                        help="Path to a pre-trained model to load and test.")

    args = parser.parse_args()
    main(args)
