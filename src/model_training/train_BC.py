
import torch
import torch.nn as nn


class BehaviorCloningGRU(nn.Module):
    def __init__(
        self,
        n_inputs=17,
        hidden_size=64,
        n_actions=3,
        dropout=0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_inputs,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        # x shape: [batch, sequence_length, 17]
        sequence_output, _ = self.gru(x)

        # Use the last observation's output
        current_representation = sequence_output[:, -1, :]

        # Predict three continuous actions
        return self.head(current_representation)


model = BehaviorCloningGRU(
    n_inputs=17,
    hidden_size=64,
    n_actions=3,
)

# Example input: 32 windows, 50 steps, 17 features
x = torch.randn(32, 50, 17)

predicted_actions = model(x)
print(predicted_actions.shape)  # [32, 3]
