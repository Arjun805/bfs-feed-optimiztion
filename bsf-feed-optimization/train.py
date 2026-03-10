"""
BSF Feed Optimization — PPO Training Script
=============================================
Trains a Stable-Baselines3 PPO agent on the BSFEnv.

Usage:
    python train.py                        # default 100k steps
    python train.py --timesteps 500000     # custom steps
    python train.py --timesteps 200000 --seed 42

Output:
    models/ppo_bsf_<timesteps>.zip    — trained model
    logs/ppo_bsf/                     — TensorBoard logs
"""

import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import BSFEnv


# ═══════════════════════════════════════════════════════════════════
# Custom callback — logs episode stats during training
# ═══════════════════════════════════════════════════════════════════
class BSFLoggingCallback(BaseCallback):
    """
    Logs BSF-specific metrics every `log_freq` episodes.
    Prints a clean progress table to the console.
    """

    def __init__(self, log_freq=50, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.biomass_history = []
        self.reward_history = []
        self.waste_history = []
        self.mortality_history = []

    def _on_step(self) -> bool:
        # Check all sub-environments for episode completions
        infos = self.locals.get("infos", [])
        for info in infos:
            if "bsf_stats" in info:
                ep = info["bsf_stats"]
                self.episode_count += 1
                self.biomass_history.append(ep["final_biomass_g"])
                self.reward_history.append(ep["r"])
                self.waste_history.append(ep["total_waste_g"])
                self.mortality_history.append(ep["total_mortality_pct"])

                if self.episode_count % self.log_freq == 0:
                    n = min(self.log_freq, len(self.biomass_history))
                    recent_biomass   = np.mean(self.biomass_history[-n:])
                    recent_reward    = np.mean(self.reward_history[-n:])
                    recent_waste     = np.mean(self.waste_history[-n:])
                    recent_mortality = np.mean(self.mortality_history[-n:])
                    best_biomass     = np.max(self.biomass_history)

                    print(
                        f"  Ep {self.episode_count:>6} | "
                        f"Biomass: {recent_biomass:>7.1f}g | "
                        f"Best: {best_biomass:>7.1f}g | "
                        f"Reward: {recent_reward:>8.1f} | "
                        f"Waste: {recent_waste:>5.1f}g | "
                        f"Mortality: {recent_mortality:>5.1f}%"
                    )
        return True

    def _on_training_end(self):
        if self.biomass_history:
            print("\n" + "=" * 65)
            print("TRAINING SUMMARY")
            print("=" * 65)
            print(f"  Total episodes     : {self.episode_count}")
            print(f"  Best biomass       : {np.max(self.biomass_history):.2f}g")
            print(f"  Avg final biomass  : {np.mean(self.biomass_history[-100:]):.2f}g")
            print(f"  Avg reward (last100): {np.mean(self.reward_history[-100:]):.2f}")
            print(f"  Avg waste  (last100): {np.mean(self.waste_history[-100:]):.2f}g")
            print(f"  Avg mortality(last100): {np.mean(self.mortality_history[-100:]):.2f}%")
            print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════
def train(timesteps=100_000, seed=42, n_envs=4):
    """
    Train a PPO agent on the BSF environment.

    Parameters
    ----------
    timesteps : int
        Total training timesteps across all environments.
    seed : int
        Random seed for reproducibility.
    n_envs : int
        Number of parallel environments for vectorized training.
    """

    print("=" * 65)
    print("BSF FEED OPTIMIZATION — PPO TRAINING")
    print("=" * 65)
    print(f"  Timesteps     : {timesteps:,}")
    print(f"  Parallel envs : {n_envs}")
    print(f"  Seed          : {seed}")
    print("=" * 65)

    # ── Create vectorized environment ────────────────────────────
    vec_env = make_vec_env(
        BSFEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"noise": True},
    )

    # ── Configure PPO ────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,              # steps per env before update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,               # discount factor
        gae_lambda=0.95,          # GAE lambda
        clip_range=0.2,
        ent_coef=0.01,            # entropy bonus (encourages exploration)
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # 2-layer MLP
        ),
        verbose=0,
        seed=seed,
        tensorboard_log="logs/",
    )

    print(f"\n  Model architecture: MLP [128, 128]")
    print(f"  Policy parameters : {sum(p.numel() for p in model.policy.parameters()):,}")
    print()

    # ── Train ────────────────────────────────────────────────────
    callback = BSFLoggingCallback(log_freq=50)

    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        tb_log_name="ppo_bsf",
        progress_bar=True,
    )
    elapsed = time.time() - start_time

    print(f"\n  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Save model ───────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    model_path = f"models/ppo_bsf_{timesteps}"
    model.save(model_path)
    print(f"  Model saved to: {model_path}.zip")

    vec_env.close()
    return model_path


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent for BSF feed optimization"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100,000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n_envs", type=int, default=4,
        help="Number of parallel environments (default: 4)"
    )
    args = parser.parse_args()

    train(timesteps=args.timesteps, seed=args.seed, n_envs=args.n_envs)
