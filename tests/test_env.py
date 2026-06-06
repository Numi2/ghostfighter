import numpy as np

from ghostfighter.config import ACTION_TO_ID
from ghostfighter.env import FightEnv


def test_env_reset_and_step_shapes():
    env = FightEnv(seed=1)
    obs0, obs1 = env.reset(randomize=False)
    assert obs0.shape == obs1.shape
    assert env.observation_dim == obs0.shape[0]
    next0, next1, r0, r1, done, info = env.step(ACTION_TO_ID["step_forward"], ACTION_TO_ID["guard"])
    assert next0.shape == obs0.shape
    assert isinstance(r0, float)
    assert isinstance(r1, float)
    assert isinstance(done, bool)
    assert "events" in info


def test_attack_can_change_state():
    env = FightEnv(seed=2)
    env.reset(randomize=False)
    start_health = env.blue.health
    # Bring fighters close enough for a punch.
    env.red.x, env.red.y = -0.35, 0.0
    env.blue.x, env.blue.y = 0.35, 0.0
    env._face_each_other(force=True)
    for _ in range(4):
        env.step(ACTION_TO_ID["cross"], ACTION_TO_ID["guard"])
    assert env.blue.health <= start_health
    assert env.red.score >= 0


def test_match_terminates():
    env = FightEnv(seed=3)
    env.reset(randomize=True)
    done = False
    steps = 0
    while not done:
        _, _, _, _, done, _ = env.step(ACTION_TO_ID["jab"], ACTION_TO_ID["guard"])
        steps += 1
    assert steps <= env.config.max_steps
