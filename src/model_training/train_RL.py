from stable_baselines3.common.logger import Logger, KVWriter, HumanOutputFormat
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import torch
import yaml, os, sys
import inspect
from datetime import datetime

import mlflow
import mlflow.onnx
import onnx
from mlflow.models import infer_signature

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from util.yaml_check import yaml_add_or_update
from util.mlflow_helpers import RL_log_code_version_tags

# Separate experiment from the VAE/dynamics-model runs, but the
# dynamics_model_run_id tag (below) keeps each RL run traceable back to the
# exact VAE run it trained against.

# UC registry: same catalog/schema as the VAE model, different model name.

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

ONNX_DYNAMIC_MODEL_PATH = config["onnx_dynamic_model_path"]
DYNAMICS_MODEL_RUN_ID = config.get("dynamics_model_run_id")

state_columns: list = config["column_config"]["state_columns"]
action_columns: list = config["column_config"]["action_columns"]
target_columns: list = config["column_config"]["target_columns"]

RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/{RAW_CSV_NAME}"


class MLflowOutputFormat(KVWriter):
    """Bridges SB3's internal logger (rollout/ep_rew_mean, train/actor_loss,
    train/critic_loss, time/fps, etc) straight into mlflow metrics."""

    def write(self, key_values, key_excluded, step=0):
        for key, value in key_values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key.replace("/", "_"), value, step=step)


class QualityLoggingCallback(BaseCallback):
    """Logs the environment's quality-closeness score at every step, as a
    second signal alongside reward -- closer to a 'residual' than the
    (exponentiated) reward itself is."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reward = 0

    def _on_step(self) -> bool:
        quality = self.training_env.get_attr("_reward")[0]
        mlflow.log_metric("quality_score", float(quality), step=self.num_timesteps)
        self.reward += quality

        return True


def train_RL(
    model,
    model_class,
    training_params: dict,
    model_directory,
    model_name,
):

    # --- bridge SB3's logger into mlflow metrics ---
    model.set_logger(
        Logger(
            folder=None,
            output_formats=[
                HumanOutputFormat(sys.stdout),
                MLflowOutputFormat(),
            ],
        )
    )

    total_timesteps = training_params["total_timesteps"]
    log_interval = training_params["log_interval"]

    qualityloggingcallback = QualityLoggingCallback()

    model.learn(
        callback=qualityloggingcallback,
        total_timesteps=total_timesteps,
        log_interval=log_interval,
    )

    average_reward = qualityloggingcallback.reward / total_timesteps

    print("model trained, model saving")

    model.save(f"{model_directory}/{model_name}")
    mlflow.log_artifact(f"{model_directory}/{model_name}.zip")

    new_model = model_class.load(f"{model_directory}/{model_name}")

    yaml_add_or_update(
        key="onnx_RL_model_path", value=f"{model_directory}/{model_name}"
    )

    # --- UC requires a signature; build one from the same dummy
    # inputs used for ONNX export plus a real forward pass ---
    return new_model, average_reward


# if __name__ == "__main__":
#     train_RL()
