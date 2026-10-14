import torch
import torch.nn as nn
import torch.optim as optim
import time


import pandas as pd
import numpy as np
from pathlib import Path

import yaml, os, sys, subprocess
import matplotlib.pyplot as plt

from src.model_classes.Dynamic.VAE_class import VAE
from src.model_training.loss_functions.mse_kld import MSEKLD

import mlflow

from mlflow.models import infer_signature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Name under which the ONNX model is registered in the MLflow Model Registry.
# This workspace uses Unity Catalog, so the registry URI and model name both
# need to be UC-flavored (3-level name: catalog.schema.model).


# ---------------------------------------------------------------------------
# get_source_hash / get_git_commit / get_git_dirty now live in
# util/mlflow_helpers.py, shared with the RL training script.
# ---------------------------------------------------------------------------


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
    state_action_target_t = x[:, 0 : x.shape[0] - 1, :]

    return state_t, state_t_1, state_action_target_t


def train_dynamic(
    model,
    loss_function,
    cv_function,
    df,
    folds,
    learning_rate,
    epochs,
    quality_weight,
    column_indexes: dict[str, int],
    state_columns: list[str],
    action_columns: list[str],
    target_columns: list[str],
    **kwargs,
):
    loader_datasets = cv_function(df=df, folds=folds, column_indexes=column_indexes, **kwargs)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # separate state(t) from state(t+1)
    state_indexes = [column_indexes[x] for x in state_columns]

    # need to make sure target(t) and target(t+1) is the same
    target_indexes = [column_indexes[x] for x in target_columns]

    # training loop

    final_val_losses = []
    run_start = time.time()

    for fold in range(folds):
        print(f"\n--- Processing Fold {fold + 1} ---")
        fold_prefix = f"fold{fold + 1}"
        fold_start = time.time()

        train_loader, val_loader, quality_train, quality_val = loader_datasets[
            f"{fold}"
        ]

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for data, _ in train_loader:
                data = data.to(device)
                state_t, state_t_1, state_action_target_t = create_state_datasets_t(
                    data, state_indexes
                )
                optimizer.zero_grad()
                predicted_delta, mu, logvar = model(state_action_target_t)
                loss = loss_function(
                    predicted_delta=predicted_delta,
                    state_t=state_t,
                    state_t_1=state_t_1,
                    mu=mu,
                    logvar=logvar,
                    quality=quality_train,
                    quality_weight=quality_weight,
                    features=model.features,
                    sequences=state_action_target_t.shape[1],
                )
                loss.backward()
                loss_item = loss.item()
                scaled_loss_item = loss_item / (
                    len(data) * state_action_target_t.shape[1] * model.features
                )
                train_loss += scaled_loss_item
                optimizer.step()
            print(f"Fold {fold + 1} - Epoch {epoch+1}, TRAIN LOSS: {train_loss}")
            mlflow.log_metric(f"{fold_prefix}/train_loss", train_loss, step=epoch)

            model.eval()
            val_loss = 0
            val_residuals_epoch = []
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    state_val_t, state_val_t_1, state_action_target_val_t = (
                        create_state_datasets_t(data, state_indexes)
                    )
                    predicted_delta, mu, logvar = model(state_action_target_val_t)
                    loss = loss_function(
                        predicted_delta=predicted_delta,
                        state_t=state_val_t,
                        state_t_1=state_val_t_1,
                        mu=mu,
                        logvar=logvar,
                        quality=quality_val,
                        quality_weight=quality_weight,
                        features=model.features,
                        sequences=state_action_target_val_t.shape[1],
                    )
                    loss_item = loss.item()
                    scaled_loss_item = loss_item / (
                        len(data) * state_action_target_val_t.shape[1] * model.features
                    )
                    val_loss += scaled_loss_item

                    # residuals = actual next-state minus predicted next-state
                    predicted_state_val_t_1 = state_val_t + predicted_delta
                    batch_residuals = (
                        (predicted_state_val_t_1 - state_val_t_1).detach().cpu().numpy()
                    )
                    val_residuals_epoch.append(batch_residuals)

            print(f"Fold {fold + 1} - Epoch {epoch+1}, VAL LOSS: {val_loss}")
            mlflow.log_metric(f"{fold_prefix}/val_loss", val_loss, step=epoch)

            if val_residuals_epoch:
                residuals_concat = np.concatenate(
                    [r.reshape(-1) for r in val_residuals_epoch]
                )
                mlflow.log_metrics(
                    {
                        f"{fold_prefix}/val_residual_mean": float(
                            residuals_concat.mean()
                        ),
                        f"{fold_prefix}/val_residual_std": float(
                            residuals_concat.std()
                        ),
                        f"{fold_prefix}/val_residual_abs_mean": float(
                            np.abs(residuals_concat).mean()
                        ),
                    },
                    step=epoch,
                )

                # only snapshot the distribution on the final epoch of the
                # fold, to avoid writing one histogram per epoch
                if epoch == epochs - 1:
                    fig, ax = plt.subplots()
                    ax.hist(residuals_concat, bins=50)
                    ax.set_title(f"Fold {fold + 1} validation residuals (final epoch)")
                    mlflow.log_figure(
                        fig, f"residual_plots/{fold_prefix}_residuals.png"
                    )
                    plt.close(fig)

        final_val_losses.append(val_loss)
        mlflow.log_metric(f"{fold_prefix}/duration_sec", time.time() - fold_start)

    # --- aggregate across folds, logged once at the run level ---

    mean_final_val_loss = float(np.mean(final_val_losses))

    mlflow.log_metrics(
        {
            "mean_final_val_loss": mean_final_val_loss,
            "std_final_val_loss": float(np.std(final_val_losses)),
            "total_train_time_sec": time.time() - run_start,
        }
    )

    return model, mean_final_val_loss
