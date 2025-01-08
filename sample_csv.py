#!/usr/bin/env python3

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
    batch_size = 1
    learning_rate = 0.0005
    num_epochs = 200
    clip_value = 1.0
    d_model = 16
    d_state = 4
    expansion_factor = 2
    num_layers = 2
    window_size = 100
    stride = 1
    label_mode = 'all'
    normalize = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset instantiation
    train_dataset = CSVDataset(csv_file=args.train_data,
                               window_size=window_size,
                               stride=stride,
                               skip_header=True,
                               label_mode=label_mode,
                               normalize=normalize)
    test_dataset = CSVDataset(csv_file=args.test_data,
                              window_size=window_size,
                              stride=stride,
                              skip_header=True,
                              label_mode=label_mode,
                              normalize=normalize)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
    input_size = train_dataset.data.shape[2]
    output_size = 1

    # Model instantiation
    model = Model(d_model,
                  d_state,
                  expansion_factor,
                  num_layers,
                  input_size,
                  output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    criterion = nn.MSELoss()

    # Training
    scaler = torch.amp.GradScaler(device.type)
    print(f"Number of parameters: {model.count_parameters()}")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward
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
        test_input, test_label = next(iter(test_dataloader))
        test_input = test_input.to(device)
        test_label = test_label.to(device)
        predicted = model(test_input).squeeze().cpu()
        predicted = torch.round(predicted)
        test_label = test_label.squeeze().cpu()

        test_data_list = test_input[0].cpu().numpy().tolist()
        predicted_list = predicted.cpu().numpy().tolist()
        actual_list = test_label.cpu().numpy().tolist()

        data = {'Test Data': test_data_list,
                'Predicted': predicted_list,
                'Actual': actual_list}
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
