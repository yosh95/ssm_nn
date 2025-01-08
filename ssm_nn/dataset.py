import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self,
                 csv_file,
                 window_size=None,
                 stride=1,
                 skip_header=False):

        df = pd.read_csv(csv_file, header=0 if skip_header else None)

        self.data = []

        if window_size is None:
            window_size = len(df)

        for i in range(0, len(df) - window_size + 1, stride):
            window_data = df[i:i + window_size]
            self.data.append(window_data)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
