"""
Script to start process plant simulation, creates first row in redis if doesn't exist
then starts both the process control and process dynamic scripts

"""

import onnxruntime as ort
import numpy as np
import yaml
import redis
from datetime import datetime
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from util.yaml_check import yaml_key_check
from util.state_input import random_action_state_target_row

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/clean_{RAW_CSV_NAME}"

HOSTNAME: str = yaml_key_check(config, "hostname") or "localhost"
REDIS_PORT: int = yaml_key_check(config, "redis_port") or 6379
ONNX_RL_MODEL_PATH: str = config["onnx_RL_model_path"]
ACTION_STREAM_NAME: str = yaml_key_check(config, "action_stream_name") or "action"
ACTION_STATE_TARGET_STREAM_NAME: str = yaml_key_check(config, "action_state_target_stream_name") or "action_state_target"
STATE_COLUMNS: list = config["column_config"]["state_columns"]
ACTION_COLUMN: list = config["column_config"]["action_columns"]
TARGET_COLUMNS: list = config["column_config"]["target_columns"]
COLUMN_INDEXES: dict = config["column_config"]["column_indexes"]

def main():
    r = redis.Redis(
        host=HOSTNAME, port=REDIS_PORT, decode_responses=True, password="reyden"
    )

    all_columns = ACTION_COLUMN + STATE_COLUMNS + TARGET_COLUMNS + ["quality"]
    action_state_target_t = random_action_state_target_row(
        csv_path=CLEAN_CSV_PATH,
        columns=all_columns,
    )


    # add data to state stream
    action_state_target_t_dict = {
        all_columns[x]: float(action_state_target_t[x]) for x in range(len(all_columns))
    }

    # time of first timestamp (make it do every 1s for now)
    import time
    count = 0 

    action_state_target_t_dict["Timestamp"] = f"{datetime.now()}"
    r.xadd(ACTION_STATE_TARGET_STREAM_NAME, action_state_target_t_dict)
    print(f"added to redis{count}")
    count += 1

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)

    except Exception as e:
        print(f"error {e}")
        sys.exit(1)
