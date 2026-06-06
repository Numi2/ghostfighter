from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .env import FightEnv


@dataclass
class VectorStep:
    obs_red: np.ndarray
    obs_blue: np.ndarray
    reward_red: np.ndarray
    reward_blue: np.ndarray
    done: np.ndarray
    info: list[dict[str, object]]


class SyncVectorFightEnv:
    """Synchronous local vector environment for batched rollout interfaces.

    This intentionally keeps the same simulator semantics as `FightEnv` while
    exposing a vectorized API. It is the local development stand-in for future
    Isaac Lab GPU vectorization.
    """

    def __init__(self, num_envs: int, config: SimConfig | None = None, seed: int = 7):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.config = config or SimConfig()
        self.envs = [FightEnv(config=SimConfig(**{**self.config.__dict__, "seed": seed + i}), seed=seed + i) for i in range(self.num_envs)]

    @property
    def observation_dim(self) -> int:
        return self.envs[0].observation_dim

    def reset(self, randomize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        red, blue = [], []
        for env in self.envs:
            obs_r, obs_b = env.reset(randomize=randomize)
            red.append(obs_r)
            blue.append(obs_b)
        return np.asarray(red, dtype=np.float32), np.asarray(blue, dtype=np.float32)

    def step(self, actions_red, actions_blue) -> VectorStep:
        red, blue, rr, rb, done, infos = [], [], [], [], [], []
        for env, ar, ab in zip(self.envs, actions_red, actions_blue):
            if env.done:
                obs_r, obs_b = env.reset(randomize=True)
                red.append(obs_r)
                blue.append(obs_b)
                rr.append(0.0)
                rb.append(0.0)
                done.append(True)
                infos.append({"auto_reset": True})
                continue
            obs_r, obs_b, r_r, r_b, is_done, info = env.step(int(ar), int(ab))
            red.append(obs_r)
            blue.append(obs_b)
            rr.append(r_r)
            rb.append(r_b)
            done.append(is_done)
            infos.append(info)
        return VectorStep(
            obs_red=np.asarray(red, dtype=np.float32),
            obs_blue=np.asarray(blue, dtype=np.float32),
            reward_red=np.asarray(rr, dtype=np.float32),
            reward_blue=np.asarray(rb, dtype=np.float32),
            done=np.asarray(done, dtype=bool),
            info=infos,
        )
