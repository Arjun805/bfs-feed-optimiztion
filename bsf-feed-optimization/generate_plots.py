"""
BSF Feed Optimization — Generate Comparative Plots
==================================================
This script runs the strategies to collect detailed trace data and 
generates high-quality comparative plots, saving them as PNG files.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from env import BSFEnv
from env.bsf_env import ACTION_MAP

from evaluate import do_nothing_policy, random_policy, rule_based_policy

# ── Setup ────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
COLORS = {"PPO Agent": "#2ca02c", "Rule-Based": "#1f77b4", "Random": "#ff7f0e", "Do-Nothing": "#d62728"}

# ── Simulation Runner ────────────────────────────────────────────────

def run_traces(env, policy_fn, n_episodes=50, name="Policy"):
    """Run episodes and collect full daily trace data."""
    episodes_summary = []
    traces = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        ep_data = {
            "strategy": name,
            "episode": ep + 1,
            "days": [0],
            "biomass": [env.sim.biomass],
            "waste": [env.sim.waste],
            "mortality": [env.sim.mortality_rate]
        }
        
        total_reward = 0.0
        
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            ep_data["days"].append(env.sim.larval_age)
            ep_data["biomass"].append(env.sim.biomass)
            ep_data["waste"].append(env.sim.waste)
            ep_data["mortality"].append(env.sim.mortality_rate)
            
        traces.append(pd.DataFrame(ep_data))
        episodes_summary.append({
            "Strategy": name,
            "Episode": ep + 1,
            "Final Biomass (g)": env.sim.biomass,
            "Total Reward": total_reward,
            "Total Waste (g)": env.sim.waste,
            "Mortality (%)": env.sim.mortality_rate
        })
        
    return pd.DataFrame(episodes_summary), pd.concat(traces, ignore_index=True)


def generate_plots(model_path, n_episodes=50):
    print(f"Running simulations ({n_episodes} episodes per strategy)...")
    env = BSFEnv(noise=True)
    
    all_summaries = []
    all_traces = []
    
    strategies = {
        "Do-Nothing": do_nothing_policy,
        "Random": random_policy,
        "Rule-Based": rule_based_policy
    }
    
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        def ppo_policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        strategies["PPO Agent"] = ppo_policy
    else:
        print(f"Warning: PPO model {model_path} not found.")

    for name, policy in strategies.items():
        summary_df, trace_df = run_traces(env, policy, n_episodes=n_episodes, name=name)
        all_summaries.append(summary_df)
        all_traces.append(trace_df)
        
    summary_data = pd.concat(all_summaries, ignore_index=True)
    trace_data = pd.concat(all_traces, ignore_index=True)
    env.close()

    print("Generating plots...")

    # 1. Boxplots for final metrics (Distribution comparison)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [("Final Biomass (g)", "Higher is better"), 
               ("Total Waste (g)", "Lower is better"), 
               ("Mortality (%)", "Lower is better")]
    
    for ax, (metric, note) in zip(axes, metrics):
        sns.boxplot(data=summary_data, x="Strategy", y=metric, palette=COLORS, ax=ax)
        ax.set_title(f"{metric}\n({note})", weight="bold")
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_performance_distributions.png"), dpi=300)
    plt.close()

    # 2. Time-Series: Biomass Growth Curves (Mean ± 95% CI)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=trace_data, x="days", y="biomass", hue="strategy", 
                 palette=COLORS, errorbar="sd", linewidth=2.5)
    plt.title("Biomass Growth Over 18-Day Cycle (Mean ± SD)", weight="bold")
    plt.xlabel("Day of Growth Cycle")
    plt.ylabel("Biomass (g)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_biomass_growth_curves.png"), dpi=300)
    plt.close()

    # 3. Efficiency Frontier: Biomass vs Waste (Scatter plot)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=summary_data, x="Total Waste (g)", y="Final Biomass (g)", 
                    hue="Strategy", palette=COLORS, alpha=0.7, s=80)
    plt.title("Efficiency Frontier: Biomass vs. Waste", weight="bold")
    # Add a visual "ideal" direction arrow
    plt.annotate("Ideal Zone\n(High Biomass, Low Waste)", 
                 xy=(5, 100), xytext=(15, 60),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_efficiency_frontier.png"), dpi=300)
    plt.close()

    # 4. Violin Plot: Reward Distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=summary_data, x="Strategy", y="Total Reward", 
                   palette=COLORS, inner="quartile")
    plt.title("Cumulative Reward Distribution Across Episodes", weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_reward_violin_plot.png"), dpi=300)
    plt.close()

    print(f"✅ Successfully saved 4 high-resolution plots to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    generate_plots(model_path="models/ppo_bsf_500000", n_episodes=50)
