import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(
        self,
        input_features,
        fc1_hidden_size,
        fc1_num_layers,
        latent_dim,
        fc4_num_layers,
        state_category_dim,
    ) -> None:
        super(VAE, self).__init__()
        self.features = input_features
        self.fc1 = nn.GRU(
            input_size=input_features,
            hidden_size=fc1_hidden_size,
            num_layers=fc1_num_layers,
            batch_first=True,
        )
        self.fc21 = nn.Linear(in_features=fc1_hidden_size, out_features=latent_dim)
        self.fc22 = nn.Linear(in_features=fc1_hidden_size, out_features=latent_dim)
        # self.fc3 = nn.Linear(in_features=latent_dim, out_features=fc1_hidden_size)
        fc4_input_size = latent_dim + input_features
        self.fc4 = nn.GRU(
            input_size=fc4_input_size,
            hidden_size=state_category_dim,
            num_layers=fc4_num_layers,
            batch_first=True,
        )
        # self.fc5 = nn.Linear(in_features=FC1_HIDDEN_SIZE, out_features=input_features)

    def encode(self, x):
        # x shape (batch_size, sequences, all features)
        _, h_1 = self.fc1(x)
        h_1 = h_1.squeeze(0)
        return self.fc21(h_1), self.fc22(h_1)

    def reparameterize(self, mu, logvar, sequence_length):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        batch_size = z.shape[0]
        latent_dim = z.shape[1]
        return z.unsqueeze(1).expand(batch_size, sequence_length, latent_dim)

    def decode(self, x, z):
        # h_3 = self.fc3(z)
        # # print(h_3.shape)
        # h_3 = h_3.unsqueeze(1)
        # # print(h_3.shape)
        # h_3 = h_3.repeat(1, sequence_length, 1)
        # # print(h_3.shape)
        out, h_4 = self.fc4(torch.cat([z, x], -1))
        return out

    def forward(self, x):
        sequence_length = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sequence_length)

        return self.decode(x=x, z=z), mu, logvar


def create_state_datasets(
    x_train_scaled: np.ndarray,
    x_val_scaled: np.ndarray,
    state_indexes: list[int],
):
    x_t_train = x_train_scaled[0 : x_train_scaled.shape[0] - 1, :]
    x_t_val = x_val_scaled[0 : x_val_scaled.shape[0] - 1, :]
    state_t_1_train = x_train_scaled[1 : x_train_scaled.shape[0], state_indexes]
    state_t_1_val = x_val_scaled[1 : x_val_scaled.shape[0], state_indexes]

    return x_t_train, x_t_val, state_t_1_train, state_t_1_val


def create_state_datasets_t(x, state_indexes: list[int]):
    index_tensor = torch.tensor(state_indexes)
    state_t = x[:, 0 : x.shape[0] - 1, index_tensor]
    state_t_1 = x[:, 1 : x.shape[0], index_tensor]
    state_action_t = x[:, 0 : x.shape[0] - 1, :]

    return state_t, state_t_1, state_action_t


def loss_function(
    predicted_delta,
    state_t,
    state_t_1,
    mu,
    logvar,
    quality: np.ndarray,
    quality_weight: float,
    features: int,
    sequences: int,
):
    # weight = np.exp(quality / quality_weight)
    # weight = weight / weight.sum()
    # weight = torch.from_numpy(weight)
    # weight = weight.view(-1, sequences, 1)
    # print(f"weight shape {weight.shape}")

    # loss = (weight * (predicted_delta - x) ** 2).mean()
    predicted_state_t_1 = state_t + predicted_delta
    MSE = nn.MSELoss(reduction="sum")(predicted_state_t_1, state_t_1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD  # (loss + KLD)


def make_windows(data_array, window_size, stride=1):
    n = (data_array.shape[0] - window_size) // stride + 1
    windows = np.stack(
        [data_array[i * stride : i * stride + window_size] for i in range(n)]
    )
    return windows


def model_export_to_onnx(model):
    model_cpu = model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, 1, model.features)
    save_directory = Path("./models")
    save_directory.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model_cpu,
        dummy_input,
        f"{str(save_directory)}/VAE_industrial",
        export_params=True,
        input_names=[f"input_name"],
        output_names=["output_name", "output_mu", "output_logvar"],
        dynamic_axes={
            "input_name": {0: "batch_size", 1: "sequence_length"},
            "output_name": {0: "batch_size", 1: "sequence_length"},
            "output_mu": {0: "batch_size"},
            "output_logvar": {0: "batch_size"},
        },
    )
    print("Model Saved")


def window_CV(
    model: VAE,
    sequences: int,
    learning_rate: float,
    stride: int,
    epochs: int,
    quality_weight: float,
    df: pd.DataFrame,
    folds: int,
    batch_size: int,
    column_indexes: dict[str, int],
    state_columns: list[str],
):

    tscv = TimeSeriesSplit(n_splits=folds)

    timestamp_index = column_indexes["Timestamp"]
    quality_index = column_indexes["quality"]
    quality_data_array = df["quality"].values
    data_array = df.drop(columns=["Timestamp", "quality"]).values

    for fold, (train_idx, val_idx) in enumerate(tscv.split(data_array)):
        print(f"\n--- Processing Fold {fold + 1} ---")

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

        # separate state(t) from state(t+1)
        state_indexes = [column_indexes[x] for x in state_columns]

        x_train_windows = make_windows(
            x_train_scaled, window_size=sequences, stride=stride
        )

        x_val_windows = make_windows(x_val_scaled, window_size=sequences, stride=stride)

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

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # training loop

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for data, _ in train_loader:
                data = data.to(device)
                state_t, state_t_1, state_action_t = create_state_datasets_t(
                    data, state_indexes
                )
                optimizer.zero_grad()
                predicted_delta, mu, logvar = model(state_action_t)
                loss = loss_function(
                    predicted_delta=predicted_delta,
                    state_t=state_t,
                    state_t_1=state_t_1,
                    mu=mu,
                    logvar=logvar,
                    quality=quality_train,
                    quality_weight=quality_weight,
                    features=model.features,
                    sequences=state_action_t.shape[1],
                )
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print(
                f"epoch {epoch+1}, TRAIN LOSS: {train_loss / (len(train_loader.dataset) * sequences * model.features)}"
            )

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    state_val_t, state_val_t_1, state_action_val_t = (
                        create_state_datasets_t(data, state_indexes)
                    )
                    predicted_delta, mu, logvar = model(state_action_val_t)
                    loss = loss_function(
                        predicted_delta=predicted_delta,
                        state_t=state_val_t,
                        state_t_1=state_val_t_1,
                        mu=mu,
                        logvar=logvar,
                        quality=quality_train,
                        quality_weight=quality_weight,
                        features=model.features,
                        sequences=state_action_t.shape[1],
                    )
                    val_loss += loss.item()

            print(
                f"epoch {epoch+1}, VAL LOSS: {val_loss / (len(val_loader.dataset) * sequences * model.features)}"
            )

    model_export_to_onnx(model)
