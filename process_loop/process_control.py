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

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

RL_MODEL_PATH = config["RL_model_path"]

session = ort.InferenceSession(RL)

# Check input/output names and shapes (useful for sanity-checking)
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)
for out in session.get_outputs():
    print(out.name, out.shape, out.type)

outputs = [x.name for x in session.get_outputs()]
inputs = [x.name for x in session.get_inputs()]

input_name = session.get_inputs()[0].name

#if last row doesnt exist in redis
    #initiate  fake state of T function from excel sheet
    action_t = session.run(
        outputs, {f"{input_name}": state_t}
    )    
    #log additonal information

    #send action_t to redis

    #continue onto loop

while 0 < 1:
        
    #wait for new row in redis

        #if new row doesn't appear for x amount of time
            # route to BC model for action_t

    action_t = session.run(
        outputs, {f"{input_name}": state_t}
    )

    #log additonal information

    #send action_t to redis

    #loop back 



