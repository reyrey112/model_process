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


def get_action(obs: dict) -> np.ndarray:
    agent_input = obs["agent"].astype(np.float32).reshape(1, -1)
    target_input = obs["target"].astype(np.float32).reshape(1, -1)
    outputs = ort_session.run(
        ["action"], {"agent": agent_input, "target": target_input}
    )
    return outputs[0][0]


