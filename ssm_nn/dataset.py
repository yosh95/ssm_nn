import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self,
                 csv_file,
                 window_size=None,
                 stride=1,
                 skip_header=False,
                 label_mode='last',
                 normalize=False):

        df = pd.read_csv(csv_file, header=0 if skip_header else None)

        label_col = df.columns[-1]
        data_cols = df.columns[:-1]

        # Normalize the data if specified
        if normalize:
            data_df = df[data_cols]
            data_np = data_df.values.astype(np.float32)
            if np.any(np.isnan(data_np)) or np.any(np.isinf(data_np)):
                data_np = np.nan_to_num(data_np,
                                        nan=0.0,
                                        posinf=0.0,
                                        neginf=0.0)
            scaler = MinMaxScaler()
            data_values = scaler.fit_transform(data_np)
        else:
            data_values = df[data_cols].values.astype(np.float32)

        label_values = df[label_col].values.astype(np.float32)

        self.data = []
        self.labels = []

        if window_size is None:
            window_size = len(df)

        for i in range(0, len(df) - window_size + 1, stride):
            window_data = data_values[i:i + window_size]
            window_labels = label_values[i:i + window_size]

            if label_mode == 'last':
                label = window_labels[-1]
            elif label_mode == 'majority':
                label = np.argmax(np.bincount(window_labels.astype(int)))
            elif label_mode == 'all':
                label = window_labels
            else:
                raise ValueError(f"Invalid label_mode: {label_mode}. " +
                                 "Must be 'last', 'majority', 'all'.")

            self.data.append(window_data)
            self.labels.append(label)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)

        if label_mode == 'all':
            self.labels = torch.tensor(np.array(self.labels),
                                       dtype=torch.float32)
        else:
            self.labels = torch.tensor(np.array(self.labels),
                                       dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
