"""
BSF Feed Optimization — Advanced Behavioral Analysis
===================================================
This script dives deep into *how* the PPO Agent outperforms
baseline strategies by analyzing its specific action distributions
and responses to environmental stressors (temperature, humidity).
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
from evaluate import rule_based_policy

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

def run_detailed_traces(model_path, n_episodes=50):
    print("Running detailed traces for PPO Agent and Rule-Based...")
    env = BSFEnv(noise=True)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model {model_path} not found.")
        return None, None
        
    model = PPO.load(model_path)
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    policies = {"PPO Agent": ppo_policy, "Rule-Based": rule_based_policy}
    all_step_data = []
    summary_data = []

    for name, policy_fn in policies.items():
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_feed_used = 0.0 # Approximation based on initial biomass and additions
            
            while not done:
                action = policy_fn(obs)
                delta_cn, delta_moist = ACTION_MAP[action]
                
                # Record state BEFORE action takes effect for correlation
                all_step_data.append({
                    "Strategy": name,
                    "Episode": ep + 1,
                    "Day": env.sim.larval_age,
                    "Biomass": env.sim.biomass,
                    "C:N": env.sim.cn_ratio,
                    "Moisture": env.sim.moisture,
                    "Temperature": env.sim.temperature,
                    "Humidity": env.sim.humidity,
                    "delta_cn": delta_cn,
                    "delta_moist": delta_moist,
                    "Action_ID": action
                })
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            # Post-episode summary
            summary_data.append({
                "Strategy": name,
                "Episode": ep + 1,
                "Final Biomass": env.sim.biomass,
                "Total Waste": env.sim.waste,
                "Biomass Yield Ratio": env.sim.biomass / (env.sim.waste + 1e-5) # Simplified proxy for FCR equivalent
            })

    env.close()
    return pd.DataFrame(all_step_data), pd.DataFrame(summary_data)

def generate_advanced_plots(step_df, summary_df):
    print("Generating advanced behavioral plots...")

    # 1. Action Distribution Comparison (Heatmap form)
    # We want to see if PPO uses a wider variety of actions or favors specific combinations
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    ppo_actions = step_df[step_df["Strategy"] == "PPO Agent"].groupby(["delta_cn", "delta_moist"]).size().reset_index(name="count")
    ppo_pivot = ppo_actions.pivot(index="delta_cn", columns="delta_moist", values="count").fillna(0)
    sns.heatmap(ppo_pivot, annot=True, fmt="g", cmap="Greens", cbar=False)
    plt.title("PPO Agent: Action Heatmap\n(Row: Δ C:N, Col: Δ Moisture)", weight="bold")
    plt.ylabel("Δ C:N Adjust")
    plt.xlabel("Δ Moisture Adjust")

    plt.subplot(1, 2, 2)
    rule_actions = step_df[step_df["Strategy"] == "Rule-Based"].groupby(["delta_cn", "delta_moist"]).size().reset_index(name="count")
    rule_pivot = rule_actions.pivot(index="delta_cn", columns="delta_moist", values="count").fillna(0)
    sns.heatmap(rule_pivot, annot=True, fmt="g", cmap="Blues", cbar=False)
    plt.title("Rule-Based: Action Heatmap", weight="bold")
    plt.ylabel("Δ C:N Adjust")
    plt.xlabel("Δ Moisture Adjust")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_action_distribution_heatmap.png"), dpi=300)
    plt.close()

    # 2. Agent Response to Temperature Stress
    # Let's see how the PPO agent adjusts moisture based on ambient temperature
    plt.figure(figsize=(8, 6))
    ppo_data = step_df[step_df["Strategy"] == "PPO Agent"].copy()
    ppo_data["Temp_Bin"] = pd.qcut(ppo_data["Temperature"], q=4, labels=["Cold", "Cool", "Warm", "Hot"])
    
    sns.barplot(data=ppo_data, x="Temp_Bin", y="delta_moist", errorbar=("ci", 95), palette="coolwarm")
    plt.title("PPO Agent: Moisture Adjustment vs. Ambient Temperature", weight="bold")
    plt.ylabel("Average Moisture Adjustment (Δ %)")
    plt.xlabel("Temperature Quartile")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.text(0.5, 0.95, "Suggests agent hydrates more when hot to combat evaporation", 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_temp_moisture_response.png"), dpi=300)
    plt.close()

    # 3. Dynamic Biological Optima Tracking (C:N Trajectory)
    # The environment requires high C:N early, low C:N late. Does the agent track this?
    plt.figure(figsize=(10, 6))
    
    # Calculate truth
    days = np.arange(0, 18)
    optimal_cn = 22.0 - (days / 18.0) * 8.0
    
    sns.lineplot(data=step_df, x="Day", y="C:N", hue="Strategy", errorbar="sd", palette={"PPO Agent": "#2ca02c", "Rule-Based": "#1f77b4"})
    plt.plot(days, optimal_cn, 'k--', linewidth=2, label="True Optimal C:N (Hidden)")
    
    plt.title("Feed Quality (C:N Ratio) Tracking Over Lifecycle", weight="bold")
    plt.ylabel("C:N Ratio")
    plt.xlabel("Day of Growth Cycle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_dynamic_optima_tracking.png"), dpi=300)
    plt.close()

    # 4. Biomass Yield Ratio (Efficiency Proxy)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=summary_df, x="Strategy", y="Biomass Yield Ratio", palette={"PPO Agent": "#2ca02c", "Rule-Based": "#1f77b4"}, errorbar="sd")
    plt.title("Feed-to-Biomass Efficiency Ratio", weight="bold")
    plt.ylabel("Biomass Generated per Unit Waste")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_efficiency_ratio.png"), dpi=300)
    plt.close()

    print(f"✅ Advanced analytical plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    step_df, summary_df = run_detailed_traces(model_path="models/ppo_bsf_500000", n_episodes=100)
    if step_df is not None:
        generate_advanced_plots(step_df, summary_df)
