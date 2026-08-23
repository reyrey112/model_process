import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import json


def z_score(col: pd.Series):
    mean = col.mean()
    std = col.std()
    z_col = col.apply(lambda x: abs(x - mean) / std)
    return z_col


datetime_format = "%m/%d/%Y %I:%M:%S %p"


def datetime_to_python(datetime_string):
    return datetime.strptime(datetime_string, datetime_format)


class CSVData:
    def __init__(self, csv_path, csv_name) -> None:
        self.csv_name = csv_name
        self.df = pd.read_csv(csv_path)

    def set_csv(self):
        pass

    def timestamp_column(self):
        for i in self.df.columns:
            remove = (
                input(f"Is this your timestamp column: '{i}'? (n): ").strip().lower()
                or "n"
            )
            if remove in ["y", "yes"]:
                self.timestamp_col = self.df[i]
                self.df = self.df.drop(columns=i)
                break

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

    def z_score_columns(self):
        self.df["state_temp_diff"] = abs(
            z_score(self.df["state_reactor_temp"])
            - z_score(self.df["action_temp_setpoint"])
        )
        self.df["state_pres_diff"] = abs(
            z_score(self.df["state_reactor_pressure"])
            - z_score(self.df["action_pressure_setpoint"])
        )
        self.df["state_yield_diff"] = abs(z_score(self.df["state_yield_pct"]) - 1)
        self.df["quality"] = (
            self.df["state_temp_diff"]
            + self.df["state_pres_diff"]
            + self.df["state_yield_diff"]
        )
        self.df.drop(
            columns=["state_temp_diff", "state_pres_diff", "state_yield_diff"],
            inplace=True,
        )

    def save_clean(self):
        self.df.to_csv(f"csvs/clean/{self.csv_name}", index=False)

    def column_ordering_config(self):
        self.df["Timestamp"] = self.timestamp_col
        column_indexes = {}
        column_names = self.df.columns
        state_columns = []
        action_columns = []
        other_columns = []
        for index, name in enumerate(column_names):
            if name.startswith("state"):
                state_columns.append(name)
            elif name.startswith("action"):
                action_columns.append(name)

            else:
                other_columns.append(name)

            column_indexes[name] = index

        column_order = []
        column_order.extend(action_columns)
        column_order.extend(state_columns)
        column_order.extend(other_columns)

        column_order_index = [column_indexes[x] for x in column_order]

        self.df = self.df.iloc[:, column_order_index]
        column_names = self.df.columns

        column_indexes = {}
        state_columns = []
        action_columns = []
        other_columns = []
        for index, name in enumerate(column_names):
            if name.startswith("state"):
                state_columns.append(name)
            elif name.startswith("action"):
                action_columns.append(name)

            else:
                other_columns.append(name)

            column_indexes[name] = index

        column_config = {
            "column_indexes": column_indexes,
            "state_columns": state_columns,
            "action_columns": action_columns,
            "other_columns": other_columns,
        }

        with open("column_config.json", "w") as file:
            json.dump(column_config, file)

    def run_pipeline(self):
        self.timestamp_column()
        self.remove_columns()
        self.encode_categorical_columns()
        self.active_state_columns()
        self.interpolate_gaps()
        self.z_score_columns()
        self.column_ordering_config()
        self.save_clean()
