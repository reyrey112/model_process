from modules.module_class import ModuleClass
import mlflow
import pandas as pd
import subprocess
from src.model_training.train_dynamic import train_dynamic
from src.model_training.training_helpers.onnx_export import dynamic_model_export_to_onnx
import yaml
from util.yaml_check import yaml_add_or_update
from util.mlflow_helpers import log_code_version_tags

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

try:
    best_loss = config["dynamic_loss"]
except:
    best_loss = None


class DynamicModule(ModuleClass):
    def __init__(self, model_class, registered_model_name, model_directory):
        self.model_class = model_class
        self.registered_model_name = registered_model_name
        self.model_directory = model_directory

    def run(
        self,
        run_name,
        training_params: dict,
        model_params: dict,
        reproducibility_enviroment_tags: dict,
        clean_csv_path,
        cv_function,
        loss_function,
        df: pd.DataFrame,
    ):

        with mlflow.start_run(run_name=run_name, log_system_metrics=True):
            log_code_version_tags(
                loss_function=loss_function, model_class=self.model_class
            )

            try:
                # --- architecture / hyperparameters ---
                mlflow.log_params(training_params)
                mlflow.log_params(model_params)

                # --- reproducibility / environment ---
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

                model = self.model_class(**model_params)

                trained_model, mean_final_val_loss = train_dynamic(
                    model=model,
                    loss_function=loss_function,
                    cv_function=cv_function,
                    df=df,
                    **training_params,
                )

                dynamic_model_export_to_onnx(
                    trained_model,
                    model_directory=self.model_directory,
                    registered_model_name=self.registered_model_name,
                )

                if best_loss is None or best_loss > mean_final_val_loss:
                    yaml_add_or_update(key="dynamic_loss", value=mean_final_val_loss)
                    result = subprocess.run(
                        ["uv", "run", "train_rl.py"], capture_output=True, text=True
                    )

            except Exception as e:
                # tag while the run is still active -- mlflow marks it FAILED
                # automatically once the exception escapes the `with` block below
                mlflow.set_tag("failure_reason", str(e))
                raise


if __name__ == "__main__":
    module = DynamicModule()
    module.run()
