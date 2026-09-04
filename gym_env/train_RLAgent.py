from stable_baselines3 import TD3
import gymnasium as gym
from ProcessSimEnv import ProcessSimEnv
import onnxruntime as ort
import pandas as pd
import torch
from pathlib import Path
import yaml, os ,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from util.yaml_check import yaml_add_or_update

class OnnxablePolicy(torch.nn.Module):
    def __init__(self, actor, action_low, action_high):
        super().__init__()
        self.actor = actor
        # register as buffers so they export as constants and move with .to(device) correctly
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, agent, target):
        obs = {"agent": agent, "target": target}
        raw_action = self.actor(obs)  # in [-1, 1]
        # rescale from [-1, 1] to [low, high]
        scaled_action = self.action_low + 0.5 * (raw_action + 1.0) * (self.action_high - self.action_low)
        return scaled_action


MODELS_FOLDER = "models"

RL_MODEL_NAME = "td3"
RL_MODEL_FOLDER = f"{MODELS_FOLDER}/RL_models"

RAW_MODEL_NAME = f"raw_{RL_MODEL_NAME}"
RAW_RL_MODEL_FOLDER = f"{RL_MODEL_FOLDER}/raw_RL_models"

ONNX_RL_MODEL_NAME = f"ONNX_{RL_MODEL_NAME}"
ONNX_RL_MODEL_FOLDER = f"{MODELS_FOLDER}/RL_ONNX"

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

ONNX_DYNAMIC_MODEL_PATH = config["onnx_dynamic_model_path"]

RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/{RAW_CSV_NAME}"

session = ort.InferenceSession(
    ONNX_DYNAMIC_MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

start_data = pd.read_csv(CLEAN_CSV_PATH)
env = gym.make("ProcessSimEnv", session=session, start_data=start_data)
print("env made, model loading")
# predicted_action_t = self.model(self.state)

model = TD3("MultiInputPolicy", env=env)
print("model loaded, learning starting")
model.learn(total_timesteps=100, log_interval=10)
print("model trained, model saving")
model.save(f"{RL_MODEL_FOLDER}/{RL_MODEL_NAME}")
print("model_saved, getting env")
vec_env = model.get_env()
print("env got, loading model")
new_model = TD3.load(f"{RL_MODEL_FOLDER}/{RL_MODEL_NAME}")
print("loaded model, env resetting")

state = vec_env.reset()

print("enc reset")

onnxable_model = OnnxablePolicy(
    new_model.policy.actor,
    action_low=env.action_space.low,
    action_high=env.action_space.high,
).to(device="cpu")
onnxable_model.actor.set_training_mode(False)
obs_shape = onnxable_model.actor.observation_space["target"].shape[0]
dummy_input = torch.randn(1, obs_shape)
dummy_input2 = torch.randn(1, obs_shape)
save_directory = Path(f"./{ONNX_RL_MODEL_FOLDER}")
save_directory.mkdir(parents=True, exist_ok=True)
onnx_RL_model_path = f"{save_directory.as_posix}/{ONNX_RL_MODEL_NAME}.onnx"
torch.onnx.export(
    onnxable_model,
    (dummy_input, dummy_input2),
    onnx_RL_model_path,
    opset_version=18,
    input_names=["agent", "target"],
    output_names=["action"],
    dynamic_axes={
        "agent": {0: "batch_size"},
        "target": {0: "batch_size"},
        "action": {0: "batch_size"},
    },
)

yaml_add_or_update(key="onnx_RL_model_path", value=onnx_RL_model_path)
print("model saved")

while True:
    action, _ = model.predict(state)
    observation, reward, done, info = vec_env.step(action)
