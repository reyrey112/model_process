from src.data import CSVData
from src.train_VAE import VAE, window_CV
import pandas as pd

import json
from pathlib import Path

CSV_PATH = "csvs/raw/chemical_process_timeseries.csv"
RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/{RAW_CSV_NAME}"
COLUMN_CONFIG_PATH = "column_config.json"

if Path("column_config.json").is_file():
    with open("column_config.json", "r") as file:
        column_configs: dict = json.load(file)
    state_columns: list = column_configs["state_columns"]
    action_columns: list = column_configs["action_columns"]
    column_indexes: dict = column_configs["column_indexes"]

FOLDS = 5
BATCH_SIZE = 40
FC1_HIDDEN_SIZE = 5
FC1_NUM_LAYERS = 1
FC4_NUM_LAYERS = 1
LATENT_DIM = 32
LEARNING_RATE = 1e-3
EPOCHS = 5
SEQUENCES = 50
STRIDE = 25
QUALITY_WEIGHT = 0.25
ENCODER_INPUT_FEATURES = len(column_indexes) - 1
DECODER_OUTPUT_STATE_FEATURES = len(state_columns) 


def main():
    # if file exists in clean csv path then option to skip
    # csv_data = CSVData(csv_path=CSV_PATH, csv_name=RAW_CSV_NAME)
    # csv_data.run_pipeline()

    df = pd.read_csv(CLEAN_CSV_PATH)

    model = VAE(
        input_features=ENCODER_INPUT_FEATURES,
        fc1_hidden_size=FC1_HIDDEN_SIZE,
        fc1_num_layers=FC1_NUM_LAYERS,
        latent_dim=LATENT_DIM,
        fc4_num_layers=FC4_NUM_LAYERS,
        state_category_dim=DECODER_OUTPUT_STATE_FEATURES,
    )

    window_CV(
        df=df,
        model=model,
        sequences=SEQUENCES,
        learning_rate=LEARNING_RATE,
        stride=STRIDE,
        epochs=EPOCHS,
        quality_weight=QUALITY_WEIGHT,
        folds=FOLDS,
        batch_size=BATCH_SIZE,
        column_indexes=column_indexes,
        state_columns=state_columns,
    )


if __name__ == "__main__":
    main()
