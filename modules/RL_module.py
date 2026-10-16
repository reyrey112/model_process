from modules.module_class import ModuleClass
from datetime import datetime
import mlflow
from util.mlflow_helpers import RL_log_code_version_tags
from util.mlflow_helpers import set_seed
import onnxruntime as ort
import gymnasium as gym
import pandas as pd
import subprocess, yaml
from src.model_training.train_RL import train_RL
from src.model_training.training_helpers.onnx_export import rl_model_export_to_onnx

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

ONNX_DYNAMIC_MODEL_PATH = config["onnx_dynamic_model_path"]


def make_run_name(total_timesteps) -> str:
    return f"td3_ts{total_timesteps}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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


class RLModule(ModuleClass):
    def __init__(
        self, model_class, registered_model_name, model_directory, model_name
    ) -> None:
        self.model_class = model_class
        self.registered_model_name = registered_model_name
        self.model_directory = model_directory
        self.model_name = model_name

    def run(
        self,
        mlflow_experiment,
        seed,
        training_params: dict,
        model_params: dict,
        reproducibility_enviroment_tags: dict,
        clean_csv_path,
        df: pd.DataFrame,
        env_class,
        onnxable_policy,
        onnx_dynamic_model_path,
        quality_distance,
        quality_closeness,
    ):

        set_seed(seed)

        mlflow.set_experiment(mlflow_experiment)

        run_name = make_run_name(total_timesteps=training_params["total_timesteps"])

        session = ort.InferenceSession(
            f"{onnx_dynamic_model_path}",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        env = gym.make(env_class.__name__, session=session, start_data=df)
        print("env made, model loading")

        model = self.model_class("MultiInputPolicy", env=env, seed=seed)
        print("model loaded, learning starting")

        model_params.update(
            {
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
            }
        )

        env_params = {
            "action_low": env.action_space.low.tolist(),
            "action_high": env.action_space.high.tolist(),
        }

        with mlflow.start_run(run_name=run_name, log_system_metrics=True):
            RL_log_code_version_tags(
                quality_distance=quality_distance,
                quality_closeness=quality_closeness,
                env_class=env_class,
                onnaxable_policy=onnxable_policy,
            )

            try:
                mlflow.log_params(training_params)
                mlflow.log_params(model_params)
                mlflow.log_params(env_params)

                mlflow.set_tags(reproducibility_enviroment_tags)

                # --- dataset lineage ---
                # mlflow.data hashes the dataframe's content itself (not just the
                # file on disk), tracks row/column counts, and shows up in the
                # run's "Datasets" tab in the UI rather than as a plain tag.
                dataset = mlflow.data.from_pandas(
                    df, source=clean_csv_path, name="clean_chemical_process_timeseries"
                )
                # takes a while can comment out to speed up
                mlflow.log_input(dataset, context="training")

                # --- dependency + config snapshots, for exact reproducibility ---
                try:
                    reqs = subprocess.check_output(["pip", "freeze"]).decode()
                    mlflow.log_text(reqs, "env_snapshots/requirements.txt")
                except Exception:
                    pass
                mlflow.log_artifact("config.yaml")

                trained_model, reward = train_RL(
                    model=model,
                    model_class=self.model_class,
                    training_params=training_params,
                    model_directory=self.model_directory,
                    model_name=self.model_name,
                )

                active_run = rl_model_export_to_onnx(
                    trained_model,
                    model_directory=self.model_directory,
                    registered_model_name=self.registered_model_name,
                    model_name=self.model_name,
                    env=env,
                    onnxable_policy=onnxable_policy,
                )

                return active_run, reward

            except Exception as e:
                # tag while the run is still active -- mlflow marks it FAILED
                # automatically once the exception escapes the `with` block below
                mlflow.set_tag("failure_reason", str(e))
                raise


# if __name__ == "__main__":
#     module = RLModule()
#     module.run()
