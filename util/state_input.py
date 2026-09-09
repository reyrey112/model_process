import pandas as pd
import os, sys
import numpy as np 

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)


def random_action_state_target_row(csv_path, columns: list):
    df = pd.read_csv(csv_path)
    action_state_target_array = df.sample()[columns].to_numpy().flatten()

    return action_state_target_array
