import pandas as pd
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)


def random_state_row(state_columns: list, csv_path):
    df = pd.read_csv(csv_path)
    state_array = df.sample()[state_columns].to_numpy().flatten()

    return state_array
