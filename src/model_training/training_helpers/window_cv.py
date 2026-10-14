import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class WindowCV:
    def __init__(self) -> None:
        pass

    def cv_function(
        self,
        df: pd.DataFrame,
        folds: int,
        sequences: int,
        stride: int,
        batch_size: int,
        column_indexes: dict[str, int],
    ):
        # Everything below logs into the single ACTIVE run that main.py already
        # opened with mlflow.start_run() -- no nested runs, metrics are
        # fold-prefixed (e.g. "fold1/train_loss") so the MLflow UI groups each
        # fold's curves while keeping one flat run to compare against others.

        windows = {}

        tscv = TimeSeriesSplit(n_splits=folds)

        timestamp_index = column_indexes["Timestamp"]
        quality_index = column_indexes["quality"]
        quality_data_array = df["quality"].values
        data_array = df.drop(columns=["Timestamp", "quality"]).values

        for fold, (train_idx, val_idx) in enumerate(tscv.split(data_array)):

            x_train_raw = data_array[train_idx]
            x_val_raw = data_array[val_idx]
            quality_train = quality_data_array[train_idx]
            quality_val = quality_data_array[val_idx]

            scaler = StandardScaler()

            x_train_scaled = scaler.fit_transform(x_train_raw)
            x_val_scaled = scaler.transform(x_val_raw)

            # re-add quality column
            x_train_scaled = np.column_stack((x_train_scaled, quality_train))
            x_val_scaled = np.column_stack((x_val_scaled, quality_val))

            x_train_windows = self.make_windows(
                x_train_scaled, window_size=sequences, stride=stride
            )

            x_val_windows = self.make_windows(
                x_val_scaled, window_size=sequences, stride=stride
            )

            train_tensor = torch.tensor(x_train_windows, dtype=torch.float32)
            val_tensor = torch.tensor(x_val_windows, dtype=torch.float32)

            train_loader = DataLoader(
                TensorDataset(train_tensor, train_tensor),
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(val_tensor, val_tensor),
                batch_size=batch_size,
                shuffle=False,
            )

            windows[f"{fold}"] = (train_loader, val_loader, quality_train, quality_val)

        return windows

    def make_windows(self, data_array, window_size, stride=1):
        n = (data_array.shape[0] - window_size) // stride + 1
        windows = np.stack(
            [data_array[i * stride : i * stride + window_size] for i in range(n)]
        )
        return windows
