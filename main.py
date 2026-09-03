from src.data import CSVData
from src.train_VAE import VAE, window_CV
import pandas as pd

import json
import yaml
from pathlib import Path

CSV_PATH = "csvs/raw/chemical_process_timeseries.csv"
RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/{RAW_CSV_NAME}"


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



def main():
    # if file exists in clean csv path then option to skip
    # csv_data = CSVData(csv_path=CSV_PATH, csv_name=RAW_CSV_NAME)
    # csv_data.run_pipeline()

    df = pd.read_csv(CLEAN_CSV_PATH)

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    state_columns: list = config["column_config"]["state_columns"]
    action_columns: list = config["column_config"]["action_columns"]
    column_indexes: dict = config["column_config"]["column_indexes"]

    ENCODER_INPUT_FEATURES = len(column_indexes) - 1
    DECODER_OUTPUT_STATE_FEATURES = len(state_columns) 

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
