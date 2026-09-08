import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import yaml, math
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from util.yaml_check import yaml_add_or_update

datetime_format = "%m/%d/%Y %I:%M:%S %p"


def datetime_to_python(datetime_string):
    return datetime.strptime(datetime_string, datetime_format)


class CSVData:
    def __init__(self, csv_path, csv_name) -> None:
        self.csv_name = csv_name
        self.df = pd.read_csv(csv_path)
        self.default = False

    def set_csv(self):
        pass

    def set_default(self):
        default = (
            input("Set Training default (for original dataset only) (y)?: ") or "y"
        )
        if default in ["y", "yes"]:
            self.default = True
            self.timestamp_col = self.df[0]
            self.df = self.df.drop(columns=0)
            self.df.drop(
                columns=["fault_type", "efficiency_loss_pct", "time_to_fault_min"]
            )

            self.encode_categorical_columns()
            self.quality_column()

            column_config = {
                "column_indexes": {
                    "action_feed_flow_rate": 0,
                    "action_coolant_flow_rate": 1,
                    "action_agitator_speed_rpm": 2,
                    "target_temp_setpoint": 3,
                    "target_pressure_setpoint": 4,
                    "state_ambient_temp_effect": 5,
                    "state_reactor_temp": 6,
                    "state_reactor_pressure": 7,
                    "state_reaction_rate": 8,
                    "state_conversion_rate": 9,
                    "state_selectivity": 10,
                    "state_yield_pct": 11,
                    "state_vibration_rms": 12,
                    "state_motor_current": 13,
                    "state_power_consumption_kw": 14,
                    "state_operating_regime_A": 15,
                    "state_operating_regime_B": 16,
                    "state_reactor_id_A_R1": 17,
                    "state_reactor_id_A_R2": 18,
                    "state_reactor_id_A_R3": 19,
                    "state_reactor_id_B_R1": 20,
                    "state_reactor_id_B_R2": 21,
                    "state_reactor_id_B_R3": 22,
                    "quality": 23,
                    "Timestamp": 24,
                },
                "state_columns": [
                    "state_ambient_temp_effect",
                    "state_reactor_temp",
                    "state_reactor_pressure",
                    "state_reaction_rate",
                    "state_conversion_rate",
                    "state_selectivity",
                    "state_yield_pct",
                    "state_vibration_rms",
                    "state_motor_current",
                    "state_power_consumption_kw",
                    "state_operating_regime_A",
                    "state_operating_regime_B",
                    "state_reactor_id_A_R1",
                    "state_reactor_id_A_R2",
                    "state_reactor_id_A_R3",
                    "state_reactor_id_B_R1",
                    "state_reactor_id_B_R2",
                    "state_reactor_id_B_R3",
                ],
                "action_columns": [
                    "action_feed_flow_rate",
                    "action_coolant_flow_rate",
                    "action_agitator_speed_rpm",
                ],
                "target_columns": [
                    "target_temp_setpoint",
                    "target_pressure_setpoint",
                ],
                "other_columns": ["quality", "Timestamp"],
            }

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

    def normalize_sampling_frequency(self):
        pass

    def split_time_sections(self):
        print("INTERPOLATE: Starting Splitting")
        split_indexes = []
        index = 0
        for i in self.timestamp_col.values:
            if i == "2024-01-01 00:00:00":
                split_indexes.append(index)
            index += 1

        else:
            split_indexes.append(index)

        self.df["Timestamp"] = self.timestamp_col

        last_index = 0
        df_splits = {}
        for i, num in enumerate(split_indexes):
            if num == 0:
                continue

            df_splits[f"df_split_{num}"] = self.df.iloc[last_index:num]
            last_index = num

        split_keys = list(df_splits.keys())
        split_dfs = {}
        for i, num in enumerate(split_keys):
            split_dfs[f"split_df_{i}"] = self.interpolate_gaps(df_splits[num])

        list_dfs = [split_dfs[x] for x in split_dfs]

        columns = list_dfs[0].columns
        column_indexes = {f"{columns[i]}": i for i, col in enumerate(columns)}
        for num, df in enumerate(list_dfs):
            for col in columns:
                value = df.iloc[0, column_indexes[col]]
                if pd.isna(value):
                    if col == "ambient_temp_effect":
                        df.iloc[0, column_indexes[col]] = 0
                    elif col == "agitator_speed_rpm" or col == "reactor_pressure":
                        sum = 0
                        count = 0
                        for i in range(10):
                            if not pd.isna(df.iloc[i + 1, column_indexes[col]]):
                                sum += df.iloc[i + 1, column_indexes[col]]
                                count += 1
                        average = sum / count
                        df.iloc[0, column_indexes[col]] = average

            list_dfs[num] = df

        self.df = pd.concat(list_dfs, ignore_index=True)
        print("INTERPOLATE: Finished")

    def interpolate_gaps(self, df):
        interpolated_df = df
        interpolated_df["Timestamp"] = pd.to_datetime(interpolated_df["Timestamp"])
        interpolated_df.set_index("Timestamp", inplace=True)

        for col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = interpolated_df[col].interpolate(method="time", limit=10)
            else:
                print(f"INTERPOLATE: Skipped column {col} because it has text.")
        return df

    def encode_categorical_columns(self):
        encoder = OneHotEncoder(sparse_output=False)
        encoder.set_output(transform="pandas")
        cols_to_drop = []
        encoded_dfs = []

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"ENCODE: Skipped column {col} because it is numerical.")
            else:
                encoded_col = encoder.fit_transform(self.df[[col]])
                encoded_dfs.append(encoded_col)
                cols_to_drop.append(col)

        if encoded_dfs:
            self.df = self.df.join(encoded_dfs).drop(columns=cols_to_drop)

    def active_state_columns(self):
        print("Label columns as Action (A), State (S), or Target/Setpoint (T)")
        for i in self.df.columns:

            column = (
                input(f"Action, State, or Target: '{i}'? (s): ").strip().lower() or "s"
            )
            if column in ["a", "action"]:
                self.df = self.df.rename(columns={i: f"action_{i}"})
            elif column in ["t", "target", "setpoint"]:
                self.df = self.df.rename(columns={i: f"target_{i}"})
            else:
                self.df = self.df.rename(columns={i: f"state_{i}"})

    def quality_column(self):
        self.df["temp_diff"] = 0 - abs(
            self.df["state_reactor_temp"] - self.df["target_temp_setpoint"]
        )
        self.df["pres_diff"] = 0 - abs(
            self.df["state_reactor_pressure"] - self.df["target_pressure_setpoint"]
        )
        self.df["yield_diff"] = abs(self.df["state_yield_pct"])
        self.df["quality"] = (
            self.df["yield_diff"] + self.df["temp_diff"] + self.df["pres_diff"]
        )
        # self.df["quality"] = self.df["quality"].apply(lambda x: math.exp(-x))
        
        self.df.drop(
            columns=["temp_diff", "pres_diff", "yield_diff"],
            inplace=True,
        )

    def column_ordering_config(self):
        self.df["Timestamp"] = self.timestamp_col
        column_indexes = {}
        column_names = self.df.columns
        state_columns = []
        action_columns = []
        target_columns = []
        other_columns = []
        for index, name in enumerate(column_names):
            if name.startswith("state"):
                state_columns.append(name)
            elif name.startswith("action"):
                action_columns.append(name)
            elif name.startswith("target"):
                target_columns.append(name)
            else:
                other_columns.append(name)

            column_indexes[name] = index

        column_order = []
        column_order.extend(action_columns)
        column_order.extend(state_columns)
        column_order.extend(target_columns)
        column_order.extend(other_columns)

        column_order_index = [column_indexes[x] for x in column_order]

        self.df = self.df.iloc[:, column_order_index]
        column_names = self.df.columns

        column_indexes = {}
        state_columns = []
        action_columns = []
        target_columns = []
        other_columns = []
        for index, name in enumerate(column_names):
            if name.startswith("state"):
                state_columns.append(name)
            elif name.startswith("action"):
                action_columns.append(name)
            elif name.startswith("target"):
                target_columns.append(name)
            else:
                other_columns.append(name)

            column_indexes[name] = index

        column_config = {
            "column_indexes": column_indexes,
            "state_columns": state_columns,
            "action_columns": action_columns,
            "target_columns": target_columns,
            "other_columns": other_columns,
        }

        yaml_add_or_update(
            yaml_file_path="config.yaml", key="column_config", value=column_config
        )

    def save_clean(self):
        self.df.to_csv(f"csvs/clean/clean_{self.csv_name}", index=False)

    def run_pipeline(self):
        # self.set_default()
        if self.default is False:
            self.timestamp_column()
            self.remove_columns()
            self.split_time_sections()
            self.encode_categorical_columns()
            self.active_state_columns()
            self.quality_column()
            self.column_ordering_config()
            self.save_clean()
