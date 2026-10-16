from src.data import CSVData
from src.model_training.train_dynamic import VAE
import pandas as pd

import yaml
from pathlib import Path

import platform
import argparse
from util.yaml_check import yaml_add_or_update

import torch
import mlflow
from mlflow import MlflowClient

from util.mlflow_helpers import set_seed

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
########## DATA PARAMETERS ##########

CSV_NAME = "chemical_process_timeseries.csv"
RAW_CSV_PATH = f"csvs/raw/{CSV_NAME}"
CLEAN_CSV_PATH = f"csvs/clean/{CSV_NAME}"


######### DYNAMIC PARAMETERS ##########
from modules.Dynamic_module import DynamicModule

from src.model_training.loss_functions.mse_kld import MSEKLD
from src.model_training.training_helpers.window_cv import WindowCV

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
DYNAMIC_SEED = 42

DYNAMIC_MODEL_NAME = "DYNAMIC_MODEL"
DYNAMIC_MLFLOW_EXPERIMENT = "/Users/reydencdavies@gmail.com/dynamic_model"
DYNAMIC_REGISTERED_MODEL_NAME = f"model_process.model_metrics.{DYNAMIC_MODEL_NAME}"
DYNAMIC_MODEL_DIRECTORY = Path("./models/pytorch_dynamic_models")


DYNAMIC_MODEL_CLASS = VAE

########## RL PARAMETERS ##########
from modules.RL_module import RLModule

from stable_baselines3 import TD3
from src.model_classes.RL.RL_class import ProcessSimEnv
from model_classes.RL.RL_class import (
    ProcessSimEnv,
    quality_distance,
    quality_closeness,
    OnnxablePolicy,
)

MODELS_FOLDER = "models"

RL_MODEL_NAME = "RL_MODEL"
RL_MODEL_FOLDER = f"{MODELS_FOLDER}/RL_models"

RAW_MODEL_NAME = f"raw_{RL_MODEL_NAME}"
RAW_RL_MODEL_FOLDER = f"{RL_MODEL_FOLDER}/raw_RL_models"

ONNX_RL_MODEL_NAME = f"ONNX_{RL_MODEL_NAME}"
ONNX_RL_MODEL_FOLDER = f"{MODELS_FOLDER}/RL_ONNX"

TOTAL_TIMESTEPS = 100
LOG_INTERVAL = 10
RL_SEED = 42
RL_MLFLOW_EXPERIMENT = "/Users/reydencdavies@gmail.com/dynamic_model_rl"
RL_REGISTERED_MODEL_NAME = f"model_process.model_metrics.{RL_MODEL_NAME}"
ONNX_DYNAMIC_MODEL_PATH = config["onnx_dynamic_model_path"]
DYNAMICS_MODEL_RUN_ID = config.get("dynamics_model_run_id")
RL_MODEL_DIRECTORY = "models/RL_models"

RL_MODEL_CLASS = TD3
RL_MODEL_ENV_CLASS = ProcessSimEnv
RL_MODEL_POLICY = "MultiInputPolicy"



def main(pipeline=0):

    # Databricks-hosted MLflow: tracking_uri="databricks" picks up auth from
    # the Databricks CLI config / DATABRICKS_HOST + DATABRICKS_TOKEN env vars.
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    client = MlflowClient()

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

    if pipeline in [0, 1]:

        model_params = {
            "model_class": f"{DYNAMIC_MODEL_CLASS}",
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
            model_class=DYNAMIC_MODEL_CLASS,
            registered_model_name=DYNAMIC_REGISTERED_MODEL_NAME,
            model_directory=DYNAMIC_MODEL_DIRECTORY,
        )

        active_run, loss = dynamic_module.run(
            mlflow_experiment=DYNAMIC_MLFLOW_EXPERIMENT,
            seed=DYNAMIC_SEED,
            training_params=training_params,
            model_params=model_params,
            reproducibility_enviroment_tags=reproducibility_enviroment_tags,
            clean_csv_path=CLEAN_CSV_PATH,
            cv_function=cv_function,
            loss_function=loss_function,
            df=df,
        )

        try:
            best_loss = config["dynamic_loss"]
        except:
            best_loss = None

        if best_loss is None or best_loss > loss:

            run_id = active_run.info.run_id

            yaml_add_or_update(key="dynamics_model_run_id", value=run_id)
            yaml_add_or_update(key="dynamic_loss", value=loss)

            matches = [
                mv
                for mv in client.search_model_versions(
                    f"name='model_process.model_metrics.{DYNAMIC_MODEL_NAME}'"
                )
                if mv.run_id == run_id
            ]
            version = max(int(mv.version) for mv in matches)
            print("setting_champion")
            client.set_registered_model_alias(f"model_process.model_metrics.{DYNAMIC_MODEL_NAME}", "champion", version)
            print("champion set")

    if pipeline in [0, 2]:

        RL_model_params = {
            "algo": "TD3",
            "policy_type": "MultiInputPolicy",
        }

        RL_training_params = {
            "total_timesteps": TOTAL_TIMESTEPS,
            "log_interval": LOG_INTERVAL,
            "seed": RL_SEED,
            "max_episode_steps": 10000,
        }

        RL_reproducibility_enviroment_tags = {
            "dynamics_model_path": ONNX_DYNAMIC_MODEL_PATH,
            "dynamics_model_run_id": DYNAMICS_MODEL_RUN_ID or "unknown",
            "state_columns": state_columns,
            "action_columns": action_columns,
            "target_columns": target_columns,
        }

        RL_module = RLModule(
            model_class=RL_MODEL_CLASS,
            model_name=RL_MODEL_NAME,
            registered_model_name=RL_REGISTERED_MODEL_NAME,
            model_directory=RL_MODEL_DIRECTORY,
        )

        active_run, reward = RL_module.run(
            mlflow_experiment=RL_MLFLOW_EXPERIMENT,
            seed=RL_SEED,
            training_params=RL_training_params,
            model_params=RL_model_params,
            reproducibility_enviroment_tags=RL_reproducibility_enviroment_tags,
            clean_csv_path=CLEAN_CSV_PATH,
            df=df,
            env_class=RL_MODEL_ENV_CLASS,
            onnxable_policy=OnnxablePolicy,
            onnx_dynamic_model_path=ONNX_DYNAMIC_MODEL_PATH,
            quality_closeness=quality_closeness,
            quality_distance=quality_distance,
        )

        try:
            best_reward = config["RL_reward"]
        except:
            best_reward = None

        if best_reward is None or best_reward > reward:

            run_id = active_run.info.run_id

            yaml_add_or_update(key="RL_model_run_id", value=run_id)
            yaml_add_or_update(key="RL_reward", value=reward)

            matches = [
                mv
                for mv in client.search_model_versions(
                    f"name='model_process.model_metrics.{RL_MODEL_NAME}'"
                )
                if mv.run_id == run_id
            ]
            version = max(int(mv.version) for mv in matches)
            print("setting champion")
            client.set_registered_model_alias(f"model_process.model_metrics.{RL_MODEL_NAME}", "champion", version)
            print("champion set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pipeline", type=int, default=0, help="which pipeline to run")

    args = parser.parse_args()

    main(pipeline=args.pipeline)
