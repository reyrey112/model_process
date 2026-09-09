import numpy as np
import onnxruntime as ort
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

ONNX_RL_MODEL_PATH = config["onnx_RL_model_path"]
ort_session = ort.InferenceSession(
    ONNX_RL_MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

agent_input = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]], dtype=np.float32)
target_input = np.array([[1]], dtype=np.float32)

outputs = ort_session.run(
    ["action"], {"agent": agent_input, "target": target_input}
)
print(outputs[0][0])


