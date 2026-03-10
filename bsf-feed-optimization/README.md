# 🪲 AI-Driven Feed Optimization for Black Soldier Fly Farming

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Environment-green.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)

> **Team Members:** [Your Team Member Names]  
> **Institution:** [Your College/University Name]  
> **Course/Subject:** [Course or Subject Name]

---

## 1. One-Line Description
A Reinforcement Learning (PPO) agent that dynamically optimizes the daily feeding strategy of Black Soldier Fly (BSF) larvae to maximize biomass and minimize waste against uncontrollable weather drift.

---

## 2. Problem Statement
Black Soldier Fly (BSF) farming is an emerging highly-efficient method for organic waste conversion and protein generation. However, current farming operations rely on **static, rule-of-thumb heuristics** for managing feed (Carbon-to-Nitrogen ratios) and moisture levels. Because larval biological needs change dynamically as they grow, and environments are subject to unpredictable temperature and humidity drift, manual methods are inherently suboptimal—leading to wasted feed, higher larval mortality, and lower overall efficiency. **BSF farming needs an AI capable of continuously adapting to real-time biological and environmental states.**

---

## 3. Solution Overview
We built a highly-realistic, 8-dimensional "digital twin" of a BSF farm (incorporating age, biomass, C:N ratio, moisture, waste, mortality, temperature, and humidity). We then trained a **Proximal Policy Optimization (PPO)** Reinforcement Learning agent to control the daily feedstock adjustments. Instead of following rigid human rules, the AI learns entirely through trial and error over millions of simulated days—discovering how to precisely track the larvae's shifting nutritional needs while preemptively combating environmental weather shocks. 

---

## 4. Architecture Diagram
```text
┌────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                │      │                 │      │                 │
│ BSF Simulator  ├──────► Gymnasium Env   ├──────►    PPO Agent    │
│ (Core Biology) │      │ (State/Actions) │      │ (RL Brain)      │
│                │      │                 │      │                 │
└────────────────┘      └─────────┬───────┘      └────────┬────────┘
                                  │                       │         
                                  ▼                       ▼         
                        ┌─────────────────┐      ┌─────────────────┐
                        │                 │      │                 │
                        │ Evaluate Script ├──────► Interactive GUI │
                        │ (Benchmarking)  │      │ (Streamlit)     │
                        │                 │      │                 │
                        └─────────────────┘      └─────────────────┘
```

---

## 5. Results
The RL Agent was evaluated against human heuristics across 50 simulated episodes with heavy environmental noise. **The PPO Agent consistently achieved higher biomass, generated less agricultural waste, and reduced larval mortality.**

| Strategy | Avg Biomass (g) | Avg Waste (g) | Avg Mortality (%) | Overall Reward |
| :--- | :--- | :--- | :--- | :--- |
| **🏆 PPO Agent** | **51.69** | **11.98** | **16.25%** | **283.17** |
| 🔵 Rule-Based | 44.61 | 13.64 | 17.55% | 220.33 |
| 🟠 Random | 21.05 | 24.02 | 32.05% | 16.49 |
| 🔴 Do-Nothing | 16.50 | 25.15 | 34.72% | -20.19 |

> *The AI agent outperformed standard farming rules by over 15% in net biomass output.*

---

## 6. Tech Stack
*   **Python 3.10+**: Core programming language.
*   **Gymnasium**: standard API wrapper for Reinforcement Learning environments.
*   **Stable-Baselines3**: state-of-the-art DRL implementations (PPO algorithms).
*   **NumPy & Pandas**: scientific computing and data manipulation.
*   **Matplotlib, Seaborn, Plotly**: 2D plotting, heatmaps, and 3D interactive visualizations.
*   **FPDF2**: Automated Executive PDF report generation.
*   **Streamlit**: web application framework for the visual dashboard.

---

## 7. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### 1. Train the Agent (Optional, a pre-trained model is included)
Train the PPO model from scratch for 500k timesteps:
```bash
python train.py --timesteps 500000 --seed 42
```

### 2. Evaluate Strategies
Run the benchmarking script to generate CLI metrics and CSV reports:
```bash
python evaluate.py --model models/ppo_bsf_500000
```

### 3. Generate Advanced Analytical Plots
Generate behavioral distribution plots (automatically saves to `results/plots/`):
```bash
python generate_plots.py
python advanced_analysis.py
```

### 4. Run the Dashboard
Launch the Interactive GUI, complete with 3D models and PDF Export tools.
```bash
streamlit run dashboard.py
```

---

## 8. Project Structure
```text
📂 bsf-feed-optimization
├── 📁 env/                   # Gymnasium RL environment bindings
│   ├── __init__.py           # Env Registry
│   └── bsf_env.py            # Space definitions (8D State, 9D Actions)
├── 📁 models/                # Saved RL weights (.zip files)
│   └── ppo_bsf_500000.zip    # Default highly-trained production model
├── 📁 results/               # CSV outputs and generated analytical plots
├── 📁 simulator/             # Core 18-day Digital Twin biological engine
│   ├── __init__.py           
│   └── bsf_simulator.py      # Growth math, temperature drift, optima tracking
├── 📁 tests/                 # Unit and sanity checks for the gym and simulator
├── advanced_analysis.py      # Generates FCR proxies and behavioral heatmaps
├── dashboard.py              # The Streamlit web application GUI
├── evaluate.py               # Benchmarks PPO against Rule-Based strategies
├── generate_plots.py         # Generates Box plots, Growth curves, etc.
├── train.py                  # PPO Model training script
└── requirements.txt          # Python dependencies
```

---

## 9. Future Work
To transition this from a simulated experiment to a production-grade controller:

1.  **Weather API Integration (Live Telemetry)**: Instead of random environmental noise, plug the simulator directly into historical or live meteorological API feeds (e.g., OpenWeatherMap) to train agents specific to regional climate seasons (Monsoon vs Dry Season).
2.  **Economic Optimization Engine**: Add real-world financial constraints to the reward function. Penalize taking actions based on labor/electricity costs and reward harvest timing dynamically (e.g., finding the mathematical point where feed cost overtakes biological growth).
3.  **Hardware Deployment (Edge AI)**: Export the `PPO` model to ONNX runtime format, allowing the "RL Brain" to run efficiently on low-power, cheap edge devices like a Raspberry Pi 4 living locally inside a BSF facility.
