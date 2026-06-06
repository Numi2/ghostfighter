from ghostfighter.config import ACTION_TO_ID
from ghostfighter.env import FightEnv
from ghostfighter.safety import CombatSafetyFirewall


def test_firewall_replaces_risky_low_balance_attack():
    env = FightEnv(seed=4)
    env.reset(randomize=False)
    env.red.balance = 0.06
    env.red.stamina = 0.08
    fw = CombatSafetyFirewall(threshold=0.50)
    decision = fw.filter(env, 0, ACTION_TO_ID["hook"])
    assert decision.overridden
    assert decision.action in {ACTION_TO_ID["recover"], ACTION_TO_ID["guard"]}
    assert decision.risk >= 0.50


def test_firewall_allows_nominal_guard():
    env = FightEnv(seed=5)
    env.reset(randomize=False)
    fw = CombatSafetyFirewall(threshold=0.80)
    decision = fw.filter(env, 0, ACTION_TO_ID["guard"])
    assert not decision.overridden
    assert decision.action == ACTION_TO_ID["guard"]
