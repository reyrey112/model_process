from stable_baselines3 import TD3
import gymnasium as gym
from ProcessSimEnv import ProcessSimEnv
import onnxruntime as ort
import pandas as pd
import torch
from pathlib import Path
import yaml


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        # actor.mu is the deterministic policy network (outputs actions directly)
        self.actor = actor

    def forward(self, agent, target):
        # Reconstruct the dict SB3's preprocess_obs expects
        obs = {"agent": agent, "target": target}
        return self.actor(obs)


MODELS_FOLDER = "models"

RL_MODEL_NAME = "td3"
RL_MODEL_FOLDER = f"{MODELS_FOLDER}/RL_models"

RAW_MODEL_NAME = f"raw_{RL_MODEL_NAME}"
RAW_RL_MODEL_FOLDER = f"{RL_MODEL_FOLDER}/raw_RL_models"

ONNX_RL_MODEL_NAME = f"ONNX_{RL_MODEL_NAME}"
ONNX_RL_MODEL_FOLDER = f"{MODELS_FOLDER}/RLONNX"

ONNX_MODEL_PATH = "C:/Users/reyde/Desktop/Coding_Project/Portfolio/model_process/models/VAE_industrial"

RAW_CSV_NAME = "chemical_process_timeseries.csv"
CLEAN_CSV_PATH = f"csvs/clean/{RAW_CSV_NAME}"

session = ort.InferenceSession(
    ONNX_MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
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

onnxable_model = OnnxablePolicy(new_model.policy.actor).to(device="cpu")
onnxable_model.actor.set_training_mode(False)
obs_shape = onnxable_model.actor.observation_space["target"].shape[0]
dummy_input = torch.randn(1, obs_shape)
dummy_input2 = torch.randn(1, obs_shape)
save_directory = Path(f"./{ONNX_RL_MODEL_FOLDER}")
save_directory.mkdir(parents=True, exist_ok=True)
onnx_RL_model_path = f"{save_directory}/{ONNX_RL_MODEL_NAME}.onnx"
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

with open("config.yaml", "w") as file:
    yaml.safe_dump({"onnx_RL_model_path": onnx_RL_model_path}, file, default_flow_style=False, sort_keys=False)


while True:
    action, _ = model.predict(state)
    observation, reward, done, info = vec_env.step(action)
