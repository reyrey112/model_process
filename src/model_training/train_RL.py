from stable_baselines3 import TD3
from stable_baselines3.common.logger import Logger, KVWriter, HumanOutputFormat
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from ProcessSimEnv import ProcessSimEnv, quality_distance, quality_closeness
import onnxruntime as ort
import pandas as pd
import numpy as np
import torch
from pathlib import Path
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
from util.mlflow_helpers import get_source_hash, get_git_commit, get_git_dirty, set_seed


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, actor, action_low, action_high):
        super().__init__()
        self.actor = actor
        # register as buffers so they export as constants and move with .to(device) correctly
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, agent, target):
        obs = {"agent": agent, "target": target}
        raw_action = self.actor(obs)  # in [-1, 1]
        # rescale from [-1, 1] to [low, high]
        scaled_action = self.action_low + 0.5 * (raw_action + 1.0) * (self.action_high - self.action_low)
        return scaled_action


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

    def _on_step(self) -> bool:
        quality = self.training_env.get_attr("_quality_score")[0]
        mlflow.log_metric("quality_score", float(quality), step=self.num_timesteps)
        return True


MODELS_FOLDER = "models"

RL_MODEL_NAME = "td3"
RL_MODEL_FOLDER = f"{MODELS_FOLDER}/RL_models"

RAW_MODEL_NAME = f"raw_{RL_MODEL_NAME}"
RAW_RL_MODEL_FOLDER = f"{RL_MODEL_FOLDER}/raw_RL_models"

ONNX_RL_MODEL_NAME = f"ONNX_{RL_MODEL_NAME}"
ONNX_RL_MODEL_FOLDER = f"{MODELS_FOLDER}/RL_ONNX"

TOTAL_TIMESTEPS = 100
LOG_INTERVAL = 10
SEED = 42

# Separate experiment from the VAE/dynamics-model runs, but the
# dynamics_model_run_id tag (below) keeps each RL run traceable back to the
# exact VAE run it trained against.
MLFLOW_EXPERIMENT_RL = "/Users/reydencdavies@gmail.com/dynamic_model_rl"

# UC registry: same catalog/schema as the VAE model, different model name.
mlflow.set_registry_uri("databricks-uc")
REGISTERED_MODEL_NAME = "model_process.model_metrics.td3_process_control"

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

ONNX_DYNAMIC_MODEL_PATH = config["onnx_dynamic_model_path"]
DYNAMICS_MODEL_RUN_ID = config.get("dynamics_model_run_id")

state_columns: list = config["column_config"]["state_columns"]
action_columns: list = config["column_config"]["action_columns"]
target_columns: list = config["column_config"]["target_columns"]

RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/{RAW_CSV_NAME}"


def safe_param(value):
    """SB3 wraps constant hyperparameters (like learning_rate) in schedule
    functions, so log their value at the start of training (progress
    remaining = 1) rather than a bare function reference."""
    if callable(value):
        try:
            return value(1)
        except Exception:
            return str(value)
    return value


def make_run_name() -> str:
    return f"td3_ts{TOTAL_TIMESTEPS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def log_code_version_tags():
    mlflow.set_tags(
        {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
            "quality_distance_hash": get_source_hash(quality_distance),
            "quality_closeness_hash": get_source_hash(quality_closeness),
            "env_step_hash": get_source_hash(ProcessSimEnv.step),
            "onnxable_policy_hash": get_source_hash(OnnxablePolicy),
        }
    )
    mlflow.log_text(inspect.getsource(ProcessSimEnv), "code_snapshots/process_sim_env.py")
    mlflow.log_text(inspect.getsource(OnnxablePolicy), "code_snapshots/onnxable_policy.py")


def main():
    set_seed(SEED)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_RL)

    session = ort.InferenceSession(
        f"{ONNX_DYNAMIC_MODEL_PATH}",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    start_data = pd.read_csv(CLEAN_CSV_PATH)
    env = gym.make("ProcessSimEnv", session=session, start_data=start_data)
    print("env made, model loading")

    model = TD3("MultiInputPolicy", env=env, seed=SEED)
    print("model loaded, learning starting")

    run_name = make_run_name()

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        try:
            log_code_version_tags()

            # --- dynamics-model lineage: which VAE run produced this env's simulator ---
            mlflow.set_tags(
                {
                    "dynamics_model_path": ONNX_DYNAMIC_MODEL_PATH,
                    "dynamics_model_run_id": DYNAMICS_MODEL_RUN_ID or "unknown",
                    "state_columns": state_columns,
                    "action_columns": action_columns,
                    "target_columns": target_columns,
                }
            )

            # --- hyperparameters, read from the live model so logged values
            # always match what actually trained, even as SB3 defaults change ---
            mlflow.log_params(
                {
                    "algo": "TD3",
                    "policy_type": "MultiInputPolicy",
                    "total_timesteps": TOTAL_TIMESTEPS,
                    "log_interval": LOG_INTERVAL,
                    "seed": SEED,
                    "learning_rate": safe_param(model.learning_rate),
                    "buffer_size": model.buffer_size,
                    "batch_size": model.batch_size,
                    "gamma": model.gamma,
                    "tau": model.tau,
                    "train_freq": str(model.train_freq),
                    "gradient_steps": model.gradient_steps,
                    "policy_delay": model.policy_delay,
                    "target_policy_noise": model.target_policy_noise,
                    "target_noise_clip": model.target_noise_clip,
                    "learning_starts": model.learning_starts,
                    "action_low": env.action_space.low.tolist(),
                    "action_high": env.action_space.high.tolist(),
                    "max_episode_steps": 10000,
                }
            )
            mlflow.log_text(str(model.policy), "code_snapshots/policy_architecture.txt")

            # --- dataset lineage for the env's starting-state sample pool ---
            dataset = mlflow.data.from_pandas(
                start_data, source=CLEAN_CSV_PATH, name="rl_start_data"
            )
            mlflow.log_input(dataset, context="env_reset_sampling")

            # --- bridge SB3's logger into mlflow metrics ---
            model.set_logger(
                Logger(
                    folder=None,
                    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
                )
            )

            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                log_interval=LOG_INTERVAL,
                callback=QualityLoggingCallback(),
            )
            print("model trained, model saving")

            model.save(f"{RL_MODEL_FOLDER}/{RL_MODEL_NAME}")
            mlflow.log_artifact(f"{RL_MODEL_FOLDER}/{RL_MODEL_NAME}.zip")
            print("model_saved, getting env")

            vec_env = model.get_env()
            print("env got, loading model")
            new_model = TD3.load(f"{RL_MODEL_FOLDER}/{RL_MODEL_NAME}")
            print("loaded model, env resetting")

            state = vec_env.reset()
            print("env reset, creating OnnxablePolicy")

            onnxable_model = OnnxablePolicy(
                new_model.policy.actor,
                action_low=env.action_space.low,
                action_high=env.action_space.high,
            ).to(device="cpu")
            onnxable_model.actor.set_training_mode(False)
            agent_shape = onnxable_model.actor.observation_space["agent"].shape[0]
            quality_shape = onnxable_model.actor.observation_space["target"].shape[0]
            agent_input = torch.randn(1, agent_shape)
            quality_input = torch.randn(1, quality_shape)
            save_directory = Path(f"./{ONNX_RL_MODEL_FOLDER}")
            save_directory.mkdir(parents=True, exist_ok=True)
            onnx_RL_model_path = f"{save_directory.as_posix()}/{ONNX_RL_MODEL_NAME}.onnx"
            print("exporting model to ONNX")
            torch.onnx.export(
                onnxable_model,
                (agent_input, quality_input),
                onnx_RL_model_path,
                opset_version=18,
                input_names=["agent", "target"],
                output_names=["action"],
                dynamic_axes={
                    "agent": {0: "batch_size"},
                    "target": {0: "batch_size"},
                    "action": {0: "batch_size"},
                },
            )

            yaml_add_or_update(key="onnx_RL_model_path", value=onnx_RL_model_path)

            # --- model size ---
            total_params = sum(p.numel() for p in onnxable_model.parameters())
            trainable_params = sum(
                p.numel() for p in onnxable_model.parameters() if p.requires_grad
            )
            mlflow.log_params(
                {"total_params": total_params, "trainable_params": trainable_params}
            )

            # --- UC requires a signature; build one from the same dummy
            # inputs used for ONNX export plus a real forward pass ---
            with torch.no_grad():
                example_action = onnxable_model(agent_input, quality_input)

            input_example = {
                "agent": agent_input.detach().cpu().numpy(),
                "target": quality_input.detach().cpu().numpy(),
            }
            output_example = example_action.detach().cpu().numpy()
            signature = infer_signature(input_example, output_example)

            onnx_model = onnx.load(onnx_RL_model_path)
            mlflow.onnx.log_model(
                onnx_model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                signature=signature,
                input_example=input_example,
            )

            print(f"model saved and registered as '{REGISTERED_MODEL_NAME}'")
        except Exception as e:
            mlflow.set_tag("failure_reason", str(e))
            raise


if __name__ == "__main__":
    main()