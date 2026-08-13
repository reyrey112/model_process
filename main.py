from src.data import CSVData
from src.train_VAE import VAE, WindowCV
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

CSV_PATH = "csvs/raw/chemical_process_timeseries.csv"
RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/{RAW_CSV_NAME}"
FOLDS = 10
BATCH_SIZE = 32

def main():
    # if file exists in clean csv path then option to skip
    # csv_data = CSVData(csv_path=CSV_PATH, csv_name=RAW_CSV_NAME)
    # csv_data.run_pipeline()
        
    window_cv = WindowCV(csv_path=CLEAN_CSV_PATH, folds = FOLDS, batch_size=BATCH_SIZE)

    window_cv.window_CV(VAE)


if __name__ == "__main__":
    main()
