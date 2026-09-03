import gymnasium as gym
from typing import Optional
import numpy as np
import torch
import pandas as pd
import onnxruntime as ort
from pathlib import Path
import json
import math

if Path("column_config.json").is_file():
    with open("column_config.json", "r") as file:
        column_configs: dict = json.load(file)
    state_columns: list = column_configs["state_columns"]
    action_columns: list = column_configs["action_columns"]
    column_indexes: dict = column_configs["column_indexes"]

timestamp_index = column_indexes["Timestamp"]
quality_index = column_indexes["quality"]


def z_score(col, x):
    mean = col.mean()
    std = col.std()
    z_score = abs(x - mean) / std
    return z_score


def quality_column(df, array: np.ndarray, indexes=column_indexes):
    temp_diff = abs(
        z_score(
            df["state_reactor_temp"],
            array[indexes["state_reactor_temp"] - len(action_columns)],
        )
        - z_score(
            df["action_temp_setpoint"],
            array[indexes["action_temp_setpoint"] - len(action_columns)],
        )
    )
    pressure_diff = abs(
        z_score(
            df["state_reactor_pressure"],
            array[indexes["state_reactor_pressure"] - len(action_columns)],
        )
        - z_score(
            df["action_pressure_setpoint"],
            array[indexes["action_pressure_setpoint"] - len(action_columns)],
        )
    )
    yield_diff = abs(
        z_score(
            df["state_yield_pct"],
            array[indexes["state_yield_pct"] - len(action_columns)],
        )
        - 1
    )
    quality = np.float32(temp_diff + pressure_diff + yield_diff)
    return quality


class ProcessSimEnv(gym.Env):

    def __init__(
        self,
        session: ort.InferenceSession,
        start_data: pd.DataFrame,
        state_columns=state_columns,
        action_columns=action_columns,
    ) -> None:
        super().__init__()

        # inference model used for providing new states
        self.session = session
        self.outputs = [x.name for x in session.get_outputs()]
        self.inputs = [x.name for x in session.get_inputs()]

        # starting dataframe for talking a real random row for inference
        self.start_data = start_data.drop(columns="Timestamp")

        # for choosing only the state columns
        self.state_indexes = [column_indexes[x] for x in state_columns]

        # initilizing locations
        self._agent_location = np.random.default_rng().random(
            len(state_columns), dtype=np.float32
        )
        self._target_location = np.random.default_rng().random(
            len(state_columns), dtype=np.float32
        )

        # bounds for action
        lower_bounds = np.array([85.0, 65.0, 250.0, 175.0, 11.0], dtype=np.float32)
        upper_bounds = np.array([115.0, 95.0, 350.0, 200.0, 15.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=lower_bounds, high=upper_bounds, shape=(5,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(
            {
                "target": gym.spaces.Box(
                    low=-50, high=400, shape=(len(state_columns),), dtype=np.float32
                ),
                "agent": gym.spaces.Box(
                    low=-50, high=400, shape=(len(state_columns),), dtype=np.float32
                ),
            },
        )

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def reset(self, seed=None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent_location = np.random.random(len(state_columns)).astype(np.float32)
        # self._agent_location = np.append(
        #     self._agent_location,
        #     quality_column(self.start_data, self._agent_location),
        # )

        self._target_location = (
            self.start_data.sample(n=1)
            .iloc[0]
            .to_numpy(dtype=np.float32)[self.state_indexes]
        )
        # self._target_location = np.append(
        #     self._target_location,
        #     quality_column(self.start_data, self._target_location),
        # )

        # .reshape(1, 1, len(state_columns))

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, predicted_action_t):
        # predicted_action_t = self.model(self.state)

        state_action_t = np.append(
            np.concatenate((predicted_action_t, self._target_location), -1),
            quality_column(self.start_data, self._target_location),
        ).reshape(1, 1, (len(state_columns) + len(action_columns))+1)

        next_state, _, _ = self.session.run(
            self.outputs, {self.inputs[0]: state_action_t}
        )
        next_state = next_state.squeeze()
        # Basic reward, large delta reward -> 0, small delta reard -> 1 for each step
        reward = math.exp(
            -abs(
                next_state[quality_index - len(action_columns) - 1]
                - self._target_location[quality_index - len(action_columns) - 1]
            )
        )

        if reward >= 100:
            terminated = True
        else:
            terminated = False

        self._target_location = next_state
        # self._target_location = np.append(
        #     self._target_location,
        #     quality_column(self.start_data, self._target_location),
        # )

        observation = self._get_obs()
        info = {}
        truncated = False
        print(f"reward: {reward}")
        return observation, reward, terminated, truncated, info


gym.register(id="ProcessSimEnv", entry_point=ProcessSimEnv, max_episode_steps=10000)
