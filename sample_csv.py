#!/usr/bin/env python3

import pdb
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from ssm_nn.dataset import CSVDataset
from ssm_nn.model import Model
from torch.utils.data import DataLoader


def main(args):
    # Hyper parameters
    batch_size = 2
    learning_rate = 0.001
    num_epochs = 1000
    clip_value = 2.0
    d_model = 16
    d_state = 2
    expansion_factor = 2
    num_layers = 2
    window_size = 6
    stride = 2
    normalize = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset instantiation
    train_dataset = CSVDataset(csv_file=args.train_data,
                               window_size=window_size,
                               stride=stride,
                               skip_header=True)
    test_dataset = CSVDataset(csv_file=args.test_data,
                              skip_header=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    input_size = torch.numel(train_dataset.data[0][0]) - 1
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
    scaler = torch.amp.GradScaler(device.type)
    print(f"Number of parameters: {model.count_parameters()}")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for i, inputs in enumerate(train_dataloader):
            train_inputs = inputs[:, :, :-1].to(device)
            train_labels = inputs[:, :, -1:].to(device).squeeze(-1)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                outputs = model(train_inputs)
                outputs = outputs.view(-1, output_size)
                labels = train_labels.view(-1).long()
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
        print(f"Epoch: {epoch+1}/{num_epochs}, " +
              f"Loss: {epoch_loss/len(train_dataloader):.4f}")

    # Test
    model.eval()
    with torch.no_grad():
        all_test_data = []
        all_predicted = []
        all_actual = []
        for inputs in test_dataloader:
           test_inputs = inputs[:, :, :-1].to(device)
           test_labels = inputs[:, :, -1:].to(device).squeeze(-1)
           logits = model(test_inputs).squeeze().cpu()
           probabilities = torch.softmax(logits, dim=1)
           predicted = torch.argmax(probabilities, dim=1)
           test_labels = test_labels.cpu().long()

           # append each data point to a single list
           # flattening if necessary
           for i in range(inputs.shape[1]): # iterate through window size (or time) dimension
               all_test_data.append(test_inputs[0,i].cpu().numpy().tolist())
               all_predicted.append(predicted[i].item())
               all_actual.append(test_labels[0,i].item())

    # Test
    model.eval()
    with torch.no_grad():
        inputs = next(iter(test_dataloader))
        test_inputs = inputs[:, :, :-1].to(device)
        test_labels = inputs[:, :, -1:].to(device).squeeze(-1)
        logits = model(test_inputs).squeeze().cpu()
        probabilities = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        test_label = test_labels.cpu()

        test_data_list = test_inputs[0].cpu().numpy().tolist()
        predicted_list = predicted.cpu().unsqueeze(-1).numpy().tolist()
        actual_list = test_label.cpu().numpy().tolist()

        data = {'Test Data': all_test_data,
                'Predicted': all_predicted,
                'Actual': all_actual}
        df = pd.DataFrame(data)
        df.to_csv(args.test_results, index=False)
        print(f"Test results saved to {args.test_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test an SSM-NN model.")
    parser.add_argument("train_data",
                        type=str,
                        help="Path to the training data CSV file.")
    parser.add_argument("test_data",
                        type=str,
                        help="Path to the test data CSV file.")
    parser.add_argument("test_results",
                        type=str,
                        help="Path to save the test results CSV file.")
    args = parser.parse_args()
    main(args)
