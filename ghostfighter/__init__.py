"""GhostFighter: autonomous robot-combat style cloning and safety gating."""

from .config import ACTION_NAMES, STYLE_NAMES, SimConfig
from .env import FightEnv

__all__ = ["ACTION_NAMES", "STYLE_NAMES", "SimConfig", "FightEnv"]
__version__ = "1.0.0"
