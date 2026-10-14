import torch
import torch.nn as nn
import numpy as np

class MSEKLD():
    def __init__(self) -> None:
        pass

    def loss_function(self,
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