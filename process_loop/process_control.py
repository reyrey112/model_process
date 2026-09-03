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

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

HOSTNAME: str = config["hostname"]
REDIS_PORT: int = config["redis_port"]
ONNX_RL_MODEL_PATH: str = config["onnx_RL_model_path"]
ACTION_STREAM_NAME: str = config["action_stream_name"]
STATE_STREAM_NAME: str = config["state_stream_name"]
STATE_COLUMNS: list = config["state_column"]
ACTION_COLUMN: list = config["action_column"]

# Client side caching using fast API for same conneciton? might not even be as fast
# as direct connect
# from redis.cache import CacheConfig

#  r = redis.Redis(
#     protocol=3,
#     cache_config=CacheConfig(),
#     decode_responses=True
# )

r = redis.Redis(host=HOSTNAME, port=REDIS_PORT, decode_responses=True)
# r = redis.Redis(
#     host="my-redis.cloud.redislabs.com", port=6379,
#     username="default", # use your Redis user. More info https://redis.io/docs/latest/operate/oss_and_stack/management/security/acl/
#     password="secret", # use your Redis password
#     ssl=True,
#     ssl_certfile="./redis_user.crt",
#     ssl_keyfile="./redis_user_private.key",
#     ssl_ca_certs="./redis_ca.pem",
# )
session = ort.InferenceSession(ONNX_RL_MODEL_PATH)

# Check input/output names and shapes (useful for sanity-checking)
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)
for out in session.get_outputs():
    print(out.name, out.shape, out.type)

outputs = [x.name for x in session.get_outputs()]
inputs = [x.name for x in session.get_inputs()]

input_name = session.get_inputs()[0].name

# check if data already exists, if not create data
latest_item = r.xrevrange(STATE_STREAM_NAME, max="+", min="-", count=1)

if latest_item:
    msg_id, state_t = latest_item[0]
else:
    # initiate state of T function from excel sheet
    state_t: list

    # time of first timestamp (make it do every 1s for now)
    time = datetime.now()

    # add data to state stream
    state_t_dict = {STATE_COLUMNS[x]: state_t[x] for x in range(len(STATE_COLUMNS))}
    state_t_dict["Timestamp"] = f"{time}"
    r.xadd(STATE_STREAM_NAME, state_t_dict)

    # get stream id for blocking
    msg_id, _ = r.xrevrange(STATE_STREAM_NAME, max="+", min="-", count=1)[0]

    # run dynamic model inference
    _, action_t, _, _ = session.run(outputs, {f"{input_name}": state_t})

    # log additonal information

    # send action_t to redis
    action_t_dict = {ACTION_COLUMNS[x]: action_t[x] for x in range(len(ACTION_COLUMNS))}
    action_t_dict["Timestamp"] = f"{time}"
    r.xadd(ACTION_STREAM_NAME, action_t_dict)

    # continue onto loop

while 0 < 1:  # adding stop conditions

    # wait for new row in redis
    try:
        latest_state_stream = r.xread(block=1000, streams={STATE_STREAM_NAME: msg_id})
        state_t_dict = latest_state_stream[0][1][1]
    except Exception as e:  # make more specific
        # if new row doesn't appear for x amount of time
        # route to BC model for action_t
        pass

    # state_t and action_t should be at same time in database
    # redis time will be used for latency checks
    time = state_t_dict.pop("Timestamp", None)

    # if time is not fonud throw error? probabaly not needed
    if time is None:
        pass

    state_t = [state_t_dict[x] for x in STATE_COLUMNS]

    action_t = session.run(outputs, {f"{input_name}": state_t})

    # log additonal information

    # send action_t to redis
    action_t_dict = {STATE_COLUMNS[x]: action_t[x] for x in range(len(STATE_COLUMNS))}
    action_t_dict["Timestamp"] = f"{time}"
    r.xadd(ACTION_STREAM_NAME, action_t_dict)

    # loop back
