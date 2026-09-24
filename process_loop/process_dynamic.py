"""
Script to simulate a manufacturing process. The script will
run a dynamic model in an onnx runtime that inputs the action(t)
from the RL model and outputs state(t+1)

"""

import onnxruntime as ort
import numpy as np
import yaml
import redis
import os, sys
from datetime import datetime
import time
from dotenv import load_dotenv


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

from util.yaml_check import yaml_key_check


HOSTNAME = os.environ.get("HOSTNAME", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

DYNAMIC_MODEL_PATH = config["onnx_dynamic_model_path"]
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
ALL_COLUMNS = ACTION_COLUMN + STATE_COLUMNS + TARGET_COLUMNS + ["quality"]
COLUMN_INDEXES: dict = config["column_config"]["column_indexes"]


def quality_distance(state_target: np.ndarray) -> np.float32:
    """
    Distance-from-ideal metric for a single state array (no action columns
    included). 0 = perfect quality, larger = worse.
    """
    temp_diff = abs(
        state_target[COLUMN_INDEXES["state_reactor_temp"]]
        - state_target[COLUMN_INDEXES["target_temp_setpoint"]]
    )

    pressure_diff = abs(
        state_target[COLUMN_INDEXES["state_reactor_pressure"]]
        - state_target[COLUMN_INDEXES["target_pressure_setpoint"]]
    )
    yield_diff = abs(state_target[COLUMN_INDEXES["state_yield_pct"]] - 1)
    return np.float32(temp_diff + pressure_diff + yield_diff)

def main ():
    session = ort.InferenceSession(f"{root_dir}/{DYNAMIC_MODEL_PATH}")

    # Check input/output names and shapes (useful for sanity-checking)
    # for inp in session.get_inputs():
    #     print(inp.name, inp.shape, inp.type)
    # for out in session.get_outputs():
    #     print(out.name, out.shape, out.type)

    # outputs = [x.name for x in session.get_outputs()]
    # inputs = [x.name for x in session.get_inputs()]

    # input_name = session.get_inputs()[0].name


    r = redis.Redis(
        host=HOSTNAME, port=int(REDIS_PORT), decode_responses=True, password=REDIS_PASSWORD
    )

    count = 0
    while 0 < 1:

        # wait for new row in redis
        try:
            latest_action_state_target_stream = r.xread(
                streams={ACTION_STATE_TARGET_STREAM_NAME: "+"}, count=1
            )

            action_state_target_t_dict = latest_action_state_target_stream[0][1][0][1]
        except Exception as e:  # make more specific
            # if new row doesn't appear for x amount of time
            # route to BC model for action_t
            pass

        action_state_target_t_array = np.array([], dtype=np.float32)
        action_state_target_t_array = np.append(
            action_state_target_t_array,
            [np.float32(action_state_target_t_dict[x]) for x in ALL_COLUMNS],
        )
        action_state_target_t_3darray = action_state_target_t_array.reshape(
            1, 1, len(ALL_COLUMNS)
        )

        t_t1_delta, mu, logvar = session.run(
            ["output_name", "output_mu", "output_logvar"],
            {"input_name": action_state_target_t_3darray},
        )

        #tiny delay between readings
        time.sleep(0.1)
        try:
            new_action = r.xread(streams={ACTION_STREAM_NAME: "$"}, count=1)
            action_t_1_dict = new_action[0][1][0][1]

        except Exception as e:
            action_t_1_dict = {
                ACTION_COLUMN[x]: float(action_state_target_t_dict[ACTION_COLUMN[x]])
                for x in range(len(ACTION_COLUMN))
            }

        state_t = action_state_target_t_array[[COLUMN_INDEXES[x] for x in STATE_COLUMNS]]
        state_t_1 = np.add(state_t, t_t1_delta).flatten()

        target_dict = {
            TARGET_COLUMNS[x]: float(action_state_target_t_dict[TARGET_COLUMNS[x]])
            for x in range(len(TARGET_COLUMNS))
        }
        state_t_1_dict = {
            STATE_COLUMNS[x]: float(state_t_1[x]) for x in range(len(STATE_COLUMNS))
        }

        quality_t_array = np.asarray(
            quality_distance(state_target=action_state_target_t_array), dtype=np.float32
        ).reshape(1, 1)

        action_state_target_t_1_dict = action_t_1_dict | state_t_1_dict | target_dict

        action_state_target_t_1_dict["quality"] = float(quality_t_array[0][0])
        action_state_target_t_1_dict["Timestamp"] = f"{datetime.now()}"

        r.xadd(ACTION_STATE_TARGET_STREAM_NAME, action_state_target_t_1_dict)

        print(f"DYNAMIC: added to redis{count}")
        count += 1

        # log mu and logvar

        # send state_t_1 to redis

        # loop back

if __name__ == "__main__":
    main()