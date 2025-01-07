import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, csv_file, batch_size, skip_header=False):
        df = pd.read_csv(csv_file, header=0 if skip_header else None)

        self.data = []
        self.labels = []

        # ラベル列は最後の列
        label_col = df.columns[-1]
        data_cols = df.columns[:-1]

        for i in range(0, batch_size):
            self.data.append(df[data_cols].values.astype(np.float32))
            self.labels.append(df[label_col].values.astype(np.float32))

        self.data = torch.tensor(np.array(self.data))
        self.labels = torch.tensor(np.array(self.labels)).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
