import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(
        self,
        input_features,
        fc1_hidden_size,
        fc1_num_layers,
        latent_dim,
        fc4_num_layers,
        state_category_dim, **kwargs
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
        sequence_length = x.shape[1]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sequence_length)

        return self.decode(x=x, z=z), mu, logvar