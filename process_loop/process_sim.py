"""
Script to simulate a manufacturing process. The script will
run a dynamic model in an onnx runtime that inputs the action(t)
from the RL model and outputs state(t+1)

"""

import onnxruntime as ort
import numpy as np
import yaml
import redis

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

DYNAMIC_MODEL_PATH = config["dynamic_model_path"]

session = ort.InferenceSession(DYNAMIC_MODEL_PATH)

# Check input/output names and shapes (useful for sanity-checking)
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)
for out in session.get_outputs():
    print(out.name, out.shape, out.type)

outputs = [x.name for x in session.get_outputs()]
inputs = [x.name for x in session.get_inputs()]

input_name = session.get_inputs()[0].name

while 0 < 1:

    #wait for new row in redis

        #if new row doesn't appear for x amount of time
            # route to BC model for action_t

    state_t_1, mu, logvar = session.run(
        outputs, {f"{input_name}": action_t}
    )

    #log mu and logvar

    #send state_t_1 to redis

    #loop back 



