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


def quality_distance(df, state: np.ndarray, indexes=column_indexes) -> np.float32:
    """
    Distance-from-ideal metric for a single state array (no action columns
    included). 0 = perfect quality, larger = worse.
    """
    temp_diff = abs(
        z_score(
            df["state_reactor_temp"],
            state[indexes["state_reactor_temp"] - len(action_columns)],
        )
        - z_score(
            df["action_temp_setpoint"],
            state[indexes["action_temp_setpoint"] - len(action_columns)],
        )
    )
    pressure_diff = abs(
        z_score(
            df["state_reactor_pressure"],
            state[indexes["state_reactor_pressure"] - len(action_columns)],
        )
        - z_score(
            df["action_pressure_setpoint"],
            state[indexes["action_pressure_setpoint"] - len(action_columns)],
        )
    )
    yield_diff = abs(
        z_score(
            df["state_yield_pct"],
            state[indexes["state_yield_pct"] - len(action_columns)],
        )
        - 1
    )
    return np.float32(temp_diff + pressure_diff + yield_diff)


def quality_closeness(df, state: np.ndarray, indexes=column_indexes) -> np.float32:
    """
    Bounded closeness score in (0, 1]. 1 = perfect quality (distance 0),
    decays toward 0 as distance grows.
    """
    dist = quality_distance(df, state, indexes)
    return np.float32(math.exp(-dist))


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

        # initilizing state and quality setpoints
        self._agent_location = np.zeros(len(state_columns), dtype=np.float32)
        self._quality_score = np.zeros(1, dtype=np.float32)


        # bounds for action
        lower_bounds = np.array([85.0, 65.0, 250.0, 175.0, 11.0], dtype=np.float32)
        upper_bounds = np.array([115.0, 95.0, 350.0, 200.0, 15.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=lower_bounds, high=upper_bounds, shape=(5,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=-50, high=400, shape=(len(state_columns),), dtype=np.float32
                ),
                "target": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
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

        self._agent_location = (
            self.start_data.sample(n=1)
            .iloc[0]
            .to_numpy(dtype=np.float32)[self.state_indexes]
        )
        self._quality_score = np.array(
            [quality_closeness(self.start_data, self._agent_location)], dtype=np.float32
        )

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, predicted_action_t):
        state_action_t = np.append(
            np.concatenate((predicted_action_t, self._agent_location), -1),
            quality_distance(self.start_data, self._agent_location),
        ).reshape(1, 1, (len(state_columns) + len(action_columns)) + 1)

        next_state, _, _ = self.session.run(self.outputs, {self.inputs[0]: state_action_t})
        next_state = next_state.squeeze()

        next_quality = np.array(
            [quality_closeness(self.start_data, next_state)], dtype=np.float32
        )

        reward = float(next_quality[0] - self._quality_score[0])

        self._agent_location = next_state
        self._quality_score = next_quality

        observation = self._get_obs()
        info = {}
        terminated = False
        truncated = False  # handled by max_episode_steps via TimeLimit wrapper

        return observation, reward, terminated, truncated, info



gym.register(id="ProcessSimEnv", entry_point=ProcessSimEnv, max_episode_steps=10000)
