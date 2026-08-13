import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FC1_HIDDEN_SIZE = 2
FC1_NUM_LAYERS = 1
FC4_NUM_LAYERS = 1
LATENT_DIM = 32
LEARNING_RATE = 1e-3
EPOCHS = 2
SEQUENCES = 500

class VAE(nn.Module):
    def __init__(self, batch_size, sequences, input_features) -> None:
        super(VAE, self).__init__()
        self.sequences = sequences
        self.input_features = input_features
        self.fc1 = nn.GRU(
            input_size=input_features,
            hidden_size=FC1_HIDDEN_SIZE,
            num_layers=FC1_NUM_LAYERS,
            batch_first=True,
        )
        self.fc21 = nn.Linear(in_features=FC1_HIDDEN_SIZE, out_features=LATENT_DIM)
        self.fc22 = nn.Linear(in_features=FC1_HIDDEN_SIZE, out_features=LATENT_DIM)
        self.fc3 = nn.Linear(in_features=LATENT_DIM, out_features=FC1_HIDDEN_SIZE)
        self.fc4 = nn.GRU(
            input_size=FC1_HIDDEN_SIZE,
            hidden_size=input_features,
            num_layers=FC4_NUM_LAYERS,
            batch_first=True,
        )
        # self.fc5 = nn.Linear(in_features=FC1_HIDDEN_SIZE, out_features=input_features)

    def encode(self, x):
        # print(x.shape)
        _, h_1 = self.fc1(x)
        # print(h_1.shape)
        h_1 = h_1.squeeze(0)
        # print(h_1.shape)
        return self.fc21(h_1), self.fc22(h_1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h_3 = self.fc3(z)
        # print(h_3.shape)
        h_3 = h_3.unsqueeze(1)
        # print(h_3.shape)
        h_3 = h_3.repeat(1,self.sequences, 1)
        # print(h_3.shape)
        out, h_4 = self.fc4(h_3)
        # print(out.shape)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        # print(mu.shape)
        # print(logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, features):
    MSE = nn.MSELoss(reduction="sum")(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE*KLD


class WindowCV:
    def __init__(self, csv_path: str, folds: int = 10, batch_size=32) -> None:
        df = pd.read_csv(csv_path)
        self.features = df.columns.tolist()
        self.data_array = df.values
        self.tscv = TimeSeriesSplit(n_splits=folds)
        self.batch_size = batch_size

    def make_windows(self, data_array,window_size):
        n = data_array.shape[0] - window_size + 1
        windows = np.stack([data_array[i:i+window_size] for i in range(n)])
        return windows

    def window_CV(self, model: VAE):

        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(self.data_array)):
            print(f"\n--- Processing Fold {fold + 1} ---")

            x_train_raw = self.data_array[train_idx]
            x_val_raw = self.data_array[val_idx]

            scaler = StandardScaler()

            x_train_scaled = scaler.fit_transform(x_train_raw)
            x_val_scaled = scaler.transform(x_val_raw)

            # fold_mean = scaler.mean_
            # fold_std = scaler.scale_
            x_train_windows = self.make_windows(x_train_scaled, window_size=SEQUENCES)
            # x_val_windows = 
            train_tensor = torch.tensor(x_train_windows, dtype=torch.float32)
            val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)

            train_loader = DataLoader(
                TensorDataset(train_tensor, train_tensor),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(val_tensor, val_tensor),
                batch_size=self.batch_size,
                shuffle=False,
            )

            # train_sequences = x_train_raw.shape[0]
            val_sequences = x_val_raw.shape[0]
            features = x_train_raw.shape[1]

            # print(f"Features being scaled: {self.features}")
            # print(f"Train samples: {len(x_train_raw)} | Val samples: {len(x_val_raw)}")

            model = VAE(self.batch_size, SEQUENCES, features)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # training loop
            model.train()
            for epoch in range(EPOCHS):
                train_loss = 0
                for batch_idx, (data, _) in enumerate(train_loader):
                    data = data.to(device)
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(data)
                    # print(recon_batch.shape)
                    loss = loss_function(recon_batch, data, mu, logvar, self.features)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
                print(f"{train_loss}")
                print(f"epoch {epoch+1}, loss: {train_loss / len(train_loader.dataset)}")

            # #eval loop
            # model.eval()
            # with torch.no_grad():
            #     val_recon, mu, logvar = model(x_val_scaled)
            #     val_loss = loss_function(val_recon, mu, logvar, self.features)
