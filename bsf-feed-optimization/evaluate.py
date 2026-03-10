"""
BSF Feed Optimization — Evaluation Script
==========================================
Compares the trained RL agent against three baseline strategies:

  1. Do-Nothing   : never adjust C:N or moisture
  2. Rule-Based   : simple if-else heuristic (tries to hit optimal)
  3. Random       : pick a random action each day
  4. PPO Agent    : trained RL policy

Usage:
    python evaluate.py                               # default model
    python evaluate.py --model models/ppo_bsf_100000 # specific model
    python evaluate.py --episodes 50                  # more episodes
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from env import BSFEnv
from env.bsf_env import ACTION_MAP


# ═══════════════════════════════════════════════════════════════════
# Baseline Strategies
# ═══════════════════════════════════════════════════════════════════

def do_nothing_policy(obs):
    """Always action 4 → (delta_cn=0, delta_moisture=0)."""
    return 4


def random_policy(obs):
    """Pick a random action from 0–8."""
    return np.random.randint(0, 9)


def rule_based_policy(obs):
    """
    Simple heuristic that tries to push C:N and moisture toward
    their biological optima (C:N=17.5, moisture=67.5%).

    Reads the normalized observation and converts back to
    approximate real values for decision-making.
    """
    # Denormalize observations
    cn_ratio  = obs[2] * 40.0    # normalized back to 10–40
    moisture  = obs[3] * 100.0   # normalized back to 30–90%

    # Decide C:N adjustment
    if cn_ratio > 19.0:
        cn_action = 0   # decrease C:N (-2)
    elif cn_ratio < 16.0:
        cn_action = 2   # increase C:N (+2)
    else:
        cn_action = 1   # hold C:N (0)

    # Decide moisture adjustment
    if moisture < 63.0:
        moist_action = 2   # increase moisture (+5)
    elif moisture > 72.0:
        moist_action = 0   # decrease moisture (-5)
    else:
        moist_action = 1   # hold moisture (0)

    # Combine into single discrete action (3×3 grid)
    # cn_action ∈ {0,1,2} → maps to rows (-2, 0, +2)
    # moist_action ∈ {0,1,2} → maps to cols (-5, 0, +5)
    return cn_action * 3 + moist_action


# ═══════════════════════════════════════════════════════════════════
# Evaluation Runner
# ═══════════════════════════════════════════════════════════════════

def run_episodes(env, policy_fn, n_episodes=20, name="Policy"):
    """
    Run multiple episodes with a given policy and collect results.

    Returns a list of dicts, one per episode.
    """
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        results.append({
            "strategy": name,
            "episode": ep + 1,
            "final_biomass_g": round(env.sim.biomass, 2),
            "total_reward": round(total_reward, 2),
            "total_waste_g": round(env.sim.waste, 2),
            "mortality_pct": round(env.sim.mortality_rate, 2),
        })

    return results


def evaluate(model_path=None, n_episodes=20):
    """
    Compare all strategies and print a summary table.
    """
    env = BSFEnv(noise=True)

    all_results = []

    # ── 1. Do-Nothing ────────────────────────────────────────────
    print("  Running Do-Nothing strategy...")
    all_results.extend(
        run_episodes(env, do_nothing_policy, n_episodes, "Do-Nothing")
    )

    # ── 2. Random ────────────────────────────────────────────────
    print("  Running Random strategy...")
    all_results.extend(
        run_episodes(env, random_policy, n_episodes, "Random")
    )

    # ── 3. Rule-Based ────────────────────────────────────────────
    print("  Running Rule-Based strategy...")
    all_results.extend(
        run_episodes(env, rule_based_policy, n_episodes, "Rule-Based")
    )

    # ── 4. PPO Agent ─────────────────────────────────────────────
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"  Running PPO Agent from {model_path}...")
        model = PPO.load(model_path)

        def ppo_policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        all_results.extend(
            run_episodes(env, ppo_policy, n_episodes, "PPO Agent")
        )
    else:
        print(f"  ⚠ No trained model found at '{model_path}' — skipping PPO.")
        print(f"    Run 'python train.py' first to train a model.\n")

    env.close()

    # ── Build comparison table ───────────────────────────────────
    df = pd.DataFrame(all_results)

    summary = df.groupby("strategy").agg(
        avg_biomass   = ("final_biomass_g", "mean"),
        std_biomass   = ("final_biomass_g", "std"),
        max_biomass   = ("final_biomass_g", "max"),
        avg_reward    = ("total_reward", "mean"),
        avg_waste     = ("total_waste_g", "mean"),
        avg_mortality = ("mortality_pct", "mean"),
    ).round(2)

    # Sort by avg biomass (best strategy on top)
    summary = summary.sort_values("avg_biomass", ascending=False)

    print("\n" + "=" * 75)
    print("STRATEGY COMPARISON")
    print("=" * 75)
    print(summary.to_string())
    print("=" * 75)

    # ── Save detailed results ────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/evaluation_results.csv", index=False)
    summary.to_csv("results/strategy_comparison.csv")
    print("\n  Detailed results saved to results/evaluation_results.csv")
    print("  Summary saved to results/strategy_comparison.csv")

    return df, summary


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate BSF feed optimization strategies"
    )
    parser.add_argument(
        "--model", type=str, default="models/ppo_bsf_100000",
        help="Path to trained model (without .zip extension)"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes per strategy (default: 20)"
    )
    args = parser.parse_args()

    print("=" * 75)
    print("BSF FEED OPTIMIZATION — STRATEGY EVALUATION")
    print("=" * 75)

    evaluate(model_path=args.model, n_episodes=args.episodes)
