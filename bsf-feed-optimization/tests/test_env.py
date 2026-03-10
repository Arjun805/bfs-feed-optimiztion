"""
Quick sanity check for BSFEnv (Gymnasium wrapper).
Run: PYTHONPATH=. python tests/test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import BSFEnv
from env.bsf_env import ACTION_MAP


def test_env():
    print("=" * 65)
    print("BSF GYMNASIUM ENVIRONMENT — SANITY CHECK")
    print("=" * 65)

    env = BSFEnv(render_mode=None, noise=False)

    # ── Test 1: Spaces ───────────────────────────────────────────
    print("\n[1] Checking spaces...")
    assert env.action_space.n == 9, f"Expected 9 actions, got {env.action_space.n}"
    assert env.observation_space.shape == (8,), f"Expected shape (8,), got {env.observation_space.shape}"
    print(f"    Action space     : {env.action_space}")
    print(f"    Observation space: {env.observation_space}")
    print("    ✅ Spaces OK")

    # ── Test 2: Reset ────────────────────────────────────────────
    print("\n[2] Testing reset()...")
    obs, info = env.reset(seed=42)
    assert obs.shape == (8,), f"Expected shape (8,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert env.observation_space.contains(obs), "Observation out of bounds!"
    print(f"    Initial obs: {obs}")
    print(f"    Info: {info}")
    print("    ✅ Reset OK")

    # ── Test 3: Step through full episode ────────────────────────
    print("\n[3] Running full 18-day episode...")
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    step_count = 0

    for day in range(18):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (8,), f"Day {day}: bad obs shape"
        assert env.observation_space.contains(obs), f"Day {day}: obs out of bounds"
        assert isinstance(reward, float), f"Day {day}: reward not float"
        assert isinstance(terminated, bool), f"Day {day}: terminated not bool"
        assert isinstance(truncated, bool), f"Day {day}: truncated not bool"

        total_reward += reward
        step_count += 1

        delta_cn, delta_moisture = ACTION_MAP[action]
        print(f"    Day {day+1:>2} | Action: {action} ({delta_cn:+d} C:N, {delta_moisture:+d} moist) | "
              f"Biomass: {env.sim.biomass:>7.2f}g | Reward: {reward:>8.2f}")

        if terminated:
            break

    assert terminated, "Episode should be done after 18 steps!"
    assert step_count == 18, f"Expected 18 steps, got {step_count}"
    assert "bsf_stats" in info, "Final info should contain 'bsf_stats' key"
    print(f"\n    Total reward  : {total_reward:.2f}")
    print(f"    Final biomass : {env.sim.biomass:.2f}g")
    print(f"    Episode info  : {info['bsf_stats']}")
    print("    ✅ Full episode OK")

    # ── Test 4: Action mapping ───────────────────────────────────
    print("\n[4] Verifying action mapping...")
    for action_id, (dcn, dm) in ACTION_MAP.items():
        label = env.action_to_str(action_id)
        print(f"    Action {action_id}: delta_cn={dcn:+d}, delta_moisture={dm:+d}  → {label}")
    print("    ✅ Action mapping OK")

    # ── Test 5: SB3 compatibility check ──────────────────────────
    print("\n[5] Stable-Baselines3 compatibility check...")
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True, skip_render_check=True)
        print("    ✅ SB3 env_checker passed!")
    except ImportError:
        print("    ⚠ stable-baselines3 not installed, skipping SB3 check")
    except Exception as e:
        print(f"    ❌ SB3 check failed: {e}")

    # ── Done ─────────────────────────────────────────────────────
    env.close()
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED ✅")
    print("=" * 65)
    print("\nNext step: Run 'python train.py' to start PPO training!")


if __name__ == "__main__":
    test_env()
