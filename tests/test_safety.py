from ghostfighter.config import ACTION_TO_ID
from ghostfighter.env import FightEnv
from ghostfighter.evaluate import run_match
from ghostfighter.policies import ScriptedPilot
from ghostfighter.safety import CombatSafetyFirewall
from ghostfighter.analysis import analyze_counterfactual_overrides


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


class FixedHookPolicy:
    name = "fixed_hook"

    def select_action(self, obs, env, fighter_idx):
        return ACTION_TO_ID["hook"]


def test_firewall_reason_counts_reset_per_match():
    env = FightEnv(seed=6)
    env.reset(randomize=False)
    env.red.balance = 0.06
    fw = CombatSafetyFirewall(threshold=0.50)
    fw.filter(env, 0, ACTION_TO_ID["hook"])
    assert fw.reason_counts
    fw.reset()
    assert fw.reason_counts == {}
    assert fw.rejections == 0


def test_counterfactual_analysis_reports_override():
    def setup(env, rng):
        env.red.x, env.red.y = -0.35, 0.0
        env.blue.x, env.blue.y = 0.35, 0.0
        env.red.balance = 0.04
        env.red.stamina = 0.08

    fw = CombatSafetyFirewall(threshold=0.50)
    _result, trace = run_match(
        FixedHookPolicy(),
        ScriptedPilot("counter", seed=9),
        seed=9,
        style_name="pressure",
        mode="firewall",
        firewall=fw,
        max_steps=8,
        collect_trace=True,
        scenario_setup=setup,
    )
    rows = analyze_counterfactual_overrides(trace, "unit", "pressure", 9)
    assert rows
    assert rows[0]["reason"] in {"low_balance", "strike_breaks_balance", "low_stamina"}
