import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class CSVData:
    def __init__(self, csv_path, csv_name) -> None:
        self.csv_name = csv_name
        self.df = pd.read_csv(csv_path)

    def set_csv(self):
        pass

    def remove_columns(self):
        answer = input("do you want to remove columns (y): ").strip().lower() or "y"
        if answer in ["y", "yes"]:
            for i in self.df.columns:
                remove = input(f"Remove '{i}'? (n): ").strip().lower() or "n"
                if remove in ["y", "yes"]:
                    self.df = self.df.drop(columns=i)

                elif remove == ["n", "no"]:
                    continue
                else:
                    continue

        elif answer in ["n", "no"]:
            return
        else:
            return

    def active_state_columns(self):
        print("Label columns as Action or State")
        for i in self.df.columns:
            column = input(f"Action or State: '{i}'? (s): ").strip().lower() or "s"
            if column in ["a", "action"]:
                self.df = self.df.rename(columns={i: f"action_{i}"})
            else:
                self.df = self.df.rename(columns={i: f"state_{i}"})

    def encode_categorical_columns(self):
        encoder = OneHotEncoder(sparse_output=False)
        encoder.set_output(transform="pandas")
        cols_to_drop = []
        encoded_dfs = []
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"Skipped column {col} because it is numerical.")
            else:
                encoded_col = encoder.fit_transform(self.df[[col]])

                encoded_dfs.append(encoded_col)
                cols_to_drop.append(col)

        if encoded_dfs:
            self.df = self.df.join(encoded_dfs).drop(columns=cols_to_drop)

    def normalize_sampling_frequency(self):
        pass

    def interpolate_gaps(self):
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].interpolate()
            else:
                print(f"Skipped column {col} because it has text.")

    def save_clean(self):
        self.df.to_csv(f"csvs/clean/{self.csv_name}", index=False)

    def run_pipeline(self):
        self.remove_columns()
        self.active_state_columns()
        self.interpolate_gaps()
        self.encode_categorical_columns()
        self.save_clean()
