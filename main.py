from src.data import CSVData
from src.model_training.train_dynamic import VAE
import pandas as pd

import os, sys
import yaml
from pathlib import Path

import platform
import subprocess
from datetime import datetime

import torch
import mlflow

from util.mlflow_helpers import set_seed
from src.model_training.loss_functions.mse_kld import MSEKLD
from src.model_training.training_helpers.window_cv import WindowCV


CSV_NAME = "chemical_process_timeseries.csv"
RAW_CSV_PATH = f"csvs/raw/{CSV_NAME}"
CLEAN_CSV_PATH = f"csvs/clean/{CSV_NAME}"


FOLDS = 2
BATCH_SIZE = 40
FC1_HIDDEN_SIZE = 5
FC1_NUM_LAYERS = 1
FC4_NUM_LAYERS = 1
LATENT_DIM = 32
LEARNING_RATE = 1e-3
EPOCHS = 2
SEQUENCES = 50
STRIDE = 25
QUALITY_WEIGHT = 0.25
SEED = 42

MLFLOW_EXPERIMENT = "/Users/reydencdavies@gmail.com/dynamic_model"
REGISTERED_MODEL_NAME = "model_process.model_metrics.dynamic_test"
MODEL_DIRECTORY = Path("./models/pytorch_dynamic_models")

MODEL_CLASS = VAE

from modules.Dynamic_module import DynamicModule


def make_run_name() -> str:
    return (
        f"vae_ld{LATENT_DIM}_lr{LEARNING_RATE}_seq{SEQUENCES}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def main():
    set_seed(SEED)

    # Databricks-hosted MLflow: tracking_uri="databricks" picks up auth from
    # the Databricks CLI config / DATABRICKS_HOST + DATABRICKS_TOKEN env vars.
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # if file exists in clean csv path then option to skip
    # csv_data = CSVData(csv_path=RAW_CSV_PATH, csv_name=RAW_CSV_NAME)
    # csv_data.run_pipeline()

    df = pd.read_csv(CLEAN_CSV_PATH)

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    state_columns: list = config["column_config"]["state_columns"]
    action_columns: list = config["column_config"]["action_columns"]
    target_columns: list = config["column_config"]["target_columns"]
    column_indexes: dict = config["column_config"]["column_indexes"]

    ENCODER_INPUT_FEATURES = len(column_indexes) - 1
    DECODER_OUTPUT_STATE_FEATURES = len(state_columns)

    run_name = make_run_name()

    model_params = {
        "input_features": ENCODER_INPUT_FEATURES,
        "fc1_hidden_size": FC1_HIDDEN_SIZE,
        "fc1_num_layers": FC1_NUM_LAYERS,
        "latent_dim": LATENT_DIM,
        "fc4_num_layers": FC4_NUM_LAYERS,
        "state_category_dim": DECODER_OUTPUT_STATE_FEATURES,
    }

    training_params = {
        "folds": FOLDS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "quality_weight": QUALITY_WEIGHT,
        "column_indexes": column_indexes,
        "state_columns": state_columns,
        "action_columns": action_columns,
        "target_columns": target_columns,
        "sequences": SEQUENCES,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
    }

    reproducibility_enviroment_tags = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    loss_function = MSEKLD().loss_function
    cv_function = WindowCV().cv_function

    dynamic_module = DynamicModule(
        model_class=MODEL_CLASS,
        registered_model_name=REGISTERED_MODEL_NAME,
        model_directory=MODEL_DIRECTORY,
    )

    dynamic_module.run(
        run_name,
        training_params=training_params,
        model_params=model_params,
        reproducibility_enviroment_tags=reproducibility_enviroment_tags,
        clean_csv_path=CLEAN_CSV_PATH,
        cv_function=cv_function,
        loss_function=loss_function,
        df=df,
    )


if __name__ == "__main__":
    main()
