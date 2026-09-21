"""
Script to simulate a control response to a manufacturing state
The script will run a RL in an onnx runtime that inputs the state(t)
from the dynamic model and outputs action(t) that the process should
take at that momemt and pass it back to the dynamic model through
redis

"""

import onnxruntime as ort
import numpy as np
import yaml
import redis
from datetime import datetime
import os, sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from util.yaml_check import yaml_key_check
from util.state_input import random_action_state_target_row

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/{RAW_CSV_NAME}"

HOSTNAME: str = yaml_key_check(config, "hostname") or "localhost"
REDIS_PORT: int = yaml_key_check(config, "redis_port") or 6379
ONNX_RL_MODEL_PATH: str = config["onnx_RL_model_path"]
STATE_TARGET_STREAM_NAME: str = (
    yaml_key_check(config, "state_target_stream_name") or "state_target"
)
ACTION_STATE_TARGET_STREAM_NAME: str = (
    yaml_key_check(config, "action_state_target_stream_name") or "action_state_target"
)
ACTION_STREAM_NAME: str = yaml_key_check(config, "action_stream_name") or "action"
STATE_COLUMNS: list = config["column_config"]["state_columns"]
ACTION_COLUMN: list = config["column_config"]["action_columns"]
TARGET_COLUMNS: list = config["column_config"]["target_columns"]
COLUMN_INDEXES: dict = config["column_config"]["column_indexes"]
STATE_TARGET_COLUMNS: list = STATE_COLUMNS + TARGET_COLUMNS


def quality_distance(state_target: np.ndarray) -> np.float32:
    """
    Distance-from-ideal metric for a single state array (no action columns
    included). 0 = perfect quality, larger = worse.
    """
    temp_diff = abs(
        state_target[COLUMN_INDEXES["state_reactor_temp"] - len(ACTION_COLUMN)]
        - state_target[COLUMN_INDEXES["target_temp_setpoint"] - len(ACTION_COLUMN)]
    )

    pressure_diff = abs(
        state_target[COLUMN_INDEXES["state_reactor_pressure"] - len(ACTION_COLUMN)]
        - state_target[COLUMN_INDEXES["target_pressure_setpoint"] - len(ACTION_COLUMN)]
    )
    yield_diff = abs(
        state_target[COLUMN_INDEXES["state_yield_pct"] - len(ACTION_COLUMN)] - 1
    )
    return np.float32(temp_diff + pressure_diff + yield_diff)

def main():

    # Client side caching using fast API for same conneciton? might not even be as fast
    # as direct connect
    # from redis.cache import CacheConfig

    #  r = redis.Redis(
    #     protocol=3,
    #     cache_config=CacheConfig(),
    #     decode_responses=True
    # )

    r = redis.Redis(
        host=HOSTNAME, port=REDIS_PORT, decode_responses=True, password="reyden"
    )
    # r = redis.Redis(
    #     host="my-redis.cloud.redislabs.com", port=6379,
    #     username="default", # use your Redis user. More info https://redis.io/docs/latest/operate/oss_and_stack/management/security/acl/
    #     password="secret", # use your Redis password
    #     ssl=True,
    #     ssl_certfile="./redis_user.crt",
    #     ssl_keyfile="./redis_user_private.key",
    #     ssl_ca_certs="./redis_ca.pem",
    # )
    session = ort.InferenceSession(f"{root_dir}/{ONNX_RL_MODEL_PATH}")

    # Check input/output names and shapes (useful for sanity-checking)
    for inp in session.get_inputs():
        print(inp.name, inp.shape, inp.type)
    for out in session.get_outputs():
        print(out.name, out.shape, out.type)

    outputs = [x.name for x in session.get_outputs()]
    inputs = [x.name for x in session.get_inputs()]

    input_name = session.get_inputs()[0].name

    count = 0
    while 0 < 1:  # adding stop conditions

        # wait for new row in redis
        try:
            latest_action_state_target_stream = r.xread(
                block=10000, streams={ACTION_STATE_TARGET_STREAM_NAME: "$"}
            )
            latest_action_state_target_dict = latest_action_state_target_stream[0][1][0][1]
        except Exception as e:  # make more specific
            # if new row doesn't appear for x amount of time
            # route to BC model for action_t
            pass

        # state_t and action_t should be at same time in database
        # redis time will be used for latency checks
        # time = state_target_t_dict.pop("Timestamp", None)

        # if time is not fonud throw error? probabaly not needed
        # if time is None:
        #     pass

        state_target_t_array = np.array([], dtype=np.float32)
        state_target_t_array = np.append(
            state_target_t_array,
            [np.float32(latest_action_state_target_dict[x]) for x in STATE_TARGET_COLUMNS],
        )

        quality_t_array = np.asarray(
            quality_distance(state_target=state_target_t_array), dtype=np.float32
        ).reshape(1, 1)
        state_target_t_array = np.array([state_target_t_array], dtype=np.float32)

        action_t = session.run(
            ["action"], {"agent": state_target_t_array, "target": quality_t_array}
        )[0][0]

        # log additonal information

        # send action_t to redis
        action_t_dict = {
            ACTION_COLUMN[x]: float(action_t[x]) for x in range(len(ACTION_COLUMN))
        }
        # action_state_target_t_dict = action_t_dict | state_target_t_dict
        # action_state_target_t_dict["quality"] = float(quality_t_array[0][0])
        # action_state_target_t_dict["Timestamp"] = f"{datetime.now()}"

        r.xadd(ACTION_STREAM_NAME, action_t_dict)
        print(f"CONTROL: added to redis{count}")
        count += 1
        # loop back

if __name__ == "__main__":
    main()