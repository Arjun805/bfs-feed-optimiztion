"""
BSF Feed Optimization — Gymnasium Environment Wrapper
======================================================
Wraps BSFSimulator into a standard gymnasium.Env so that
Stable-Baselines3 (PPO, A2C, etc.) can plug in directly.

Action Space  : Discrete(9)
    A 3×3 grid of (delta_cn, delta_moisture) combinations:
        0 → (-2, -5)   1 → (-2,  0)   2 → (-2, +5)
        3 → ( 0, -5)   4 → ( 0,  0)   5 → ( 0, +5)
        6 → (+2, -5)   7 → (+2,  0)   8 → (+2, +5)

Observation Space : Box(8,)  float32 in [0, 1]
    [norm_age, norm_biomass, norm_cn, norm_moisture,
     norm_mortality, norm_waste, norm_temp, norm_humid]

Episode Length : 18 steps (one per day of the growth cycle)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator import BSFSimulator


# ── Action lookup table ──────────────────────────────────────────────
# Maps discrete action index → (delta_cn, delta_moisture)
ACTION_MAP = {
    0: (-2, -5),
    1: (-2,  0),
    2: (-2, +5),
    3: ( 0, -5),
    4: ( 0,  0),
    5: ( 0, +5),
    6: (+2, -5),
    7: (+2,  0),
    8: (+2, +5),
}


class BSFEnv(gym.Env):
    """
    OpenAI Gymnasium environment for BSF larval feed optimization.

    The agent's goal is to learn a feeding strategy (adjusting C:N ratio
    and moisture daily) that maximizes final biomass at harvest (day 18)
    while minimizing waste and larval mortality.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, render_mode=None, noise=True):
        """
        Parameters
        ----------
        render_mode : str or None
            "human" prints to console each step; None is silent.
        noise : bool
            If True, simulator adds ±10% biological randomness.
        """
        super().__init__()

        self.render_mode = render_mode
        self.noise = noise

        # ── Spaces ───────────────────────────────────────────────
        # 9 discrete actions (3 C:N choices × 3 moisture choices)
        self.action_space = spaces.Discrete(9)

        # 8-dim observation, all normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # ── Internal simulator ───────────────────────────────────
        self.sim = BSFSimulator(noise=self.noise)

        # Episode tracking (useful for logging callbacks)
        self.episode_reward = 0.0
        self.episode_length = 0

    # ─────────────────────────────────────────────────────────────
    # Standard Gymnasium API
    # ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset the environment to day 0 of a new growth cycle.

        Returns
        -------
        observation : np.ndarray  shape (6,)
        info        : dict
        """
        super().reset(seed=seed)

        # Seed numpy RNG for reproducibility during training
        if seed is not None:
            np.random.seed(seed)

        obs = self.sim.reset()
        self.episode_reward = 0.0
        self.episode_length = 0

        info = {
            "day": 0,
            "biomass_g": self.sim.biomass,
            "cn_ratio": self.sim.cn_ratio,
            "moisture_pct": self.sim.moisture,
            "temperature_c": self.sim.temperature,
            "humidity_pct": self.sim.humidity,
        }

        if self.render_mode == "human":
            self.sim.render()

        return obs, info

    def step(self, action):
        """
        Execute one day of the growth cycle.

        Parameters
        ----------
        action : int  (0–8, index into ACTION_MAP)

        Returns
        -------
        observation : np.ndarray  shape (6,)
        reward      : float
        terminated  : bool   (True when 18-day cycle ends)
        truncated   : bool   (always False — no time limit beyond cycle)
        info        : dict   (detailed metrics for logging)
        """
        # Decode discrete action → continuous adjustments
        delta_cn, delta_moisture = ACTION_MAP[int(action)]

        # Step the simulator
        obs, reward, done, info = self.sim.step(delta_cn, delta_moisture)

        # Track episode stats
        self.episode_reward += reward
        self.episode_length += 1

        # Add episode-level info for SB3 callbacks
        if done:
            info["bsf_stats"] = {
                "r": round(self.episode_reward, 2),
                "l": self.episode_length,
                "final_biomass_g": round(self.sim.biomass, 2),
                "total_waste_g": round(self.sim.waste, 2),
                "total_mortality_pct": round(self.sim.mortality_rate, 2),
            }

        if self.render_mode == "human":
            self.sim.render()

        # Gymnasium API: (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render current state (called automatically if render_mode='human')."""
        if self.render_mode == "ansi":
            return (
                f"Day {self.sim.larval_age:>2} | "
                f"Biomass: {self.sim.biomass:>7.2f}g | "
                f"C:N: {self.sim.cn_ratio:>5.1f} | "
                f"Moisture: {self.sim.moisture:>5.1f}% | "
                f"Waste: {self.sim.waste:>6.2f}g | "
                f"Mortality: {self.sim.mortality_rate:>5.2f}% | "
                f"Temp: {self.sim.temperature:>4.1f}C | "
                f"Hum: {self.sim.humidity:>4.1f}%"
            )
        elif self.render_mode == "human":
            self.sim.render()

    def close(self):
        """Clean up (nothing to do for this env)."""
        pass

    # ─────────────────────────────────────────────────────────────
    # Utility methods
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def action_to_str(action):
        """Human-readable description of an action."""
        delta_cn, delta_moisture = ACTION_MAP[action]
        cn_str = f"C:N {delta_cn:+d}" if delta_cn != 0 else "C:N  0"
        moist_str = f"Moist {delta_moisture:+d}" if delta_moisture != 0 else "Moist  0"
        return f"[{cn_str}, {moist_str}]"
