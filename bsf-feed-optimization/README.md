# 🪲 AI-Driven Feed Optimization for Black Soldier Fly Farming

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Team Members:** Anton Gilchrist A · Dev Arjun G · Jabin Joseph M
> **Institution:** Loyola-ICAM College of Engineering and Technology (LICET)
> **Course:** PDL — Project Driven Learning

---

## 📌 Table of Contents
1. [What This Project Does](#1-what-this-project-does)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [How It Works — Full Architecture](#4-how-it-works--full-architecture)
5. [The Simulator — Digital Twin](#5-the-simulator--digital-twin)
6. [The RL Agent — Why PPO](#6-the-rl-agent--why-ppo)
7. [Results](#7-results)
8. [Tech Stack](#8-tech-stack)
9. [How to Run](#9-how-to-run)
10. [Project Structure](#10-project-structure)
11. [Future Work](#11-future-work)

---

## 1. What This Project Does

A **Reinforcement Learning (PPO) agent** that learns to optimally manage the daily feeding conditions of Black Soldier Fly (BSF) larvae — adjusting feed Carbon-to-Nitrogen ratio and substrate moisture — to **maximize biomass yield** while minimizing waste and mortality across an 18-day growth cycle, even under unpredictable temperature and humidity drift.

In simple terms: **we replaced a human farm manager's daily decisions with a self-learning AI.**

---

## 2. Problem Statement

Black Soldier Fly farming is one of the most efficient methods for converting organic food waste into high-protein animal feed. Despite this promise, current operations suffer from a fundamental limitation:

**Feeding decisions are made using fixed, static rules** — "keep C:N between 15–20, keep moisture at 65%" — regardless of the larvae's actual growth stage or the day's environmental conditions.

This fails for three reasons:

- **Larval needs shift dynamically.** Young larvae need high-carbon feed for energy. Mature larvae need high-protein feed for biomass. A static rule cannot track this.
- **Environments are unpredictable.** Temperature and humidity drift daily. Substrate dries out at different rates. No fixed rule can adapt to this.
- **Effects are delayed.** A bad feeding decision on Day 5 may not show up as mortality until Day 8. Rule-based systems cannot anticipate this.

The result: wasted feed, unnecessary larval mortality, and suboptimal harvest yields at scale.

---

## 3. Solution Overview

We modeled BSF larval farming as a **Reinforcement Learning problem** — the same framework used to train AIs to play chess, control robots, and manage data center cooling.

```
The AI acts as a farm manager.
Every day it observes the farm's state (8 variables)
and decides: adjust feed C:N? adjust moisture?
It learns through thousands of simulated 18-day cycles
what decisions lead to the best harvest.
```

Key design choices:

- **8-dimensional state space** — age, biomass, C:N ratio, moisture, waste, mortality, temperature, humidity
- **9 discrete actions** — all combinations of C:N adjustment (−2, 0, +2) × moisture adjustment (−5, 0, +5)
- **Dynamic biological optima** — optimal C:N and moisture shift with larval age, not fixed values
- **Environmental noise** — temperature and humidity drift randomly each day, simulating real farm unpredictability
- **PPO algorithm** — chosen for stability, discrete action compatibility, and fast convergence

---

## 4. How It Works — Full Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                │
│                                                                 │
│  ┌──────────────────┐    8D state     ┌──────────────────┐     │
│  │  BSF Simulator   │ ─────────────► │   Gymnasium Env  │     │
│  │  (Digital Twin)  │                │   (State/Reward) │     │
│  │                  │ ◄───────────── │                  │     │
│  │  • Growth math   │  delta_cn      └────────┬─────────┘     │
│  │  • Temp drift    │  delta_moisture          │               │
│  │  • Mortality     │                          │ obs + reward  │
│  │  • Waste calc    │                          ▼               │
│  └──────────────────┘               ┌──────────────────┐      │
│                                     │   PPO Agent      │      │
│                                     │   (Neural Net)   │      │
│                                     │   8→[128,128]→9  │      │
│                                     └────────┬─────────┘      │
│                                              │                 │
│                          ┌───────────────────┴──────────┐     │
│                          ▼                              ▼     │
│               ┌──────────────────┐          ┌──────────────┐  │
│               │  evaluate.py     │          │ dashboard.py │  │
│               │  Benchmarking    │          │  Streamlit   │  │
│               │  vs baselines    │          │  Visual Demo │  │
│               └──────────────────┘          └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Each file's role:**

| File | Role | What it does |
|---|---|---|
| `simulator/bsf_simulator.py` | Digital Twin | 18-day BSF growth math with temperature/humidity drift |
| `env/bsf_env.py` | Gym Wrapper | Translates simulator into standard RL interface |
| `train.py` | Training | Runs PPO across 500k timesteps, saves model |
| `evaluate.py` | Benchmarking | Compares PPO vs Rule-Based vs Random vs Do-Nothing |
| `advanced_analysis.py` | Deep Analysis | Action heatmaps, temperature response, efficiency ratios |
| `dashboard.py` | Visual Demo | Streamlit web app showing all results interactively |

---

## 5. The Simulator — Digital Twin

The simulator is the foundation of the project. It mathematically models how BSF larvae respond to feeding and environmental conditions each day.

### State Variables (8 dimensions)

| Variable | Meaning | Range |
|---|---|---|
| `larval_age` | Day of growth cycle | 0 → 18 |
| `biomass` | Total larval weight in grams | starts 10g |
| `cn_ratio` | Carbon:Nitrogen ratio of feed | 10 → 40 |
| `moisture` | Substrate moisture % | 30 → 90% |
| `temperature` | Ambient temperature °C | 15 → 40°C |
| `humidity` | Ambient relative humidity % | 30 → 100% |
| `mortality_rate` | Cumulative % of larvae dead | 0 → 100% |
| `waste` | Unconsumed feed in grams | 0 → ∞ |

### Dynamic Biological Optima
Unlike fixed-rule systems, the simulator's optima shift with larval age:

```python
optimal_cn       = 22.0 - (age / 18) × 8.0
# Day 0:  C:N = 22  (high carbon — energy for young larvae)
# Day 18: C:N = 14  (high protein — biomass for mature larvae)

optimal_moisture = 75.0 - (age / 18) × 15.0
# Day 0:  75%  (wet — young larvae need moisture)
# Day 18: 60%  (drier — prepupation requires drier substrate)
```

### Environmental Drift (Uncontrollable)
Every day, external conditions change randomly — the agent must **adapt**, not just follow rules:
```
Temperature: ±1.5°C per day
Humidity:    ±3.0% per day
Moisture:    −0.5 to −2.5% per day (evaporation)
C:N ratio:   ±0.5 to +1.0 per day (natural feed degradation)
```

---

## 6. The RL Agent — Why PPO

### What is PPO?
Proximal Policy Optimization (PPO) is a policy gradient RL algorithm that learns by directly improving its action-selection policy. Its key innovation is a **clipped objective function** that prevents catastrophically large brain updates during training — making it stable even in noisy biological environments.

### Why PPO over alternatives?

| Algorithm | Action Type | Stability | Fits BSF? | Reason |
|---|---|---|---|---|
| **PPO** ✅ | Discrete | High | ✅ **Yes** | Stable, fast, discrete-compatible |
| DQN | Discrete | Medium | ⚠️ Okay | Works but slower convergence |
| A2C | Discrete | Medium | ⚠️ Okay | PPO strictly outperforms A2C |
| SAC | **Continuous only** | Very High | ❌ No | Wrong action space type |
| TD3 | **Continuous only** | High | ❌ No | Wrong action space type |
| DDPG | **Continuous only** | Low | ❌ No | Unstable + wrong type |

**Three reasons PPO is the right choice:**
1. **Problem fit** — 9 discrete actions, 18-step episodes → exactly PPO's design profile
2. **Noise tolerance** — biological randomness (±10%) doesn't destabilize PPO's clipped updates
3. **Practicality** — converges in ~15 min on CPU, reproducible with fixed seed, battle-tested in academic literature

### Neural Network Architecture
```
Input: 8 neurons (one per state variable)
       ↓
Hidden Layer 1: 128 neurons (ReLU)
       ↓
Hidden Layer 2: 128 neurons (ReLU)
       ↓
Output: 9 neurons (one per possible action)
        → softmax → pick best action
```

---

## 7. Results

The PPO Agent was evaluated against three baseline strategies across **50 simulated episodes** with full environmental noise (temperature drift, humidity fluctuation, substrate evaporation).

| Strategy | Avg Biomass (g) | Avg Waste (g) | Avg Mortality (%) | Avg Reward |
|:---|:---:|:---:|:---:|:---:|
| 🏆 **PPO Agent** | **51.69** | **11.98** | **16.25%** | **283.17** |
| 🔵 Rule-Based | 44.61 | 13.64 | 17.55% | 220.33 |
| 🟠 Random | 21.05 | 24.02 | 32.05% | 16.49 |
| 🔴 Do-Nothing | 16.50 | 25.15 | 34.72% | −20.19 |

**Key takeaways:**
- PPO outperforms Rule-Based by **+15.9% biomass** despite both having access to the same information
- PPO generates **12.2% less waste** than Rule-Based — it uses feed more efficiently
- PPO achieves **7.4% lower mortality** than Rule-Based — it better adapts to temperature stress
- Rule-Based collapses relative to PPO specifically when **temperature drift** pushes conditions away from its hardcoded thresholds — PPO adapts, rules cannot

---

## 8. Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Gymnasium | 0.29+ | RL environment standard API |
| Stable-Baselines3 | 2.0+ | PPO implementation |
| NumPy | 1.24+ | Array math, biological calculations |
| Pandas | 2.0+ | Results aggregation and CSV export |
| Matplotlib | 3.7+ | Growth curves, comparison plots |
| Seaborn | 0.12+ | Action heatmaps, behavioral analysis |
| Streamlit | latest | Interactive web dashboard |
| FPDF2 | latest | PDF report export |

---

## 9. How to Run

### Prerequisites
```bash
git clone https://github.com/Arjun805/bfs-feed-optimiztion.git
cd bfs-feed-optimiztion
pip install -r requirements.txt
```

### Step 1 — Verify Simulator (Day 1 sanity check)
```bash
python tests/test_simulator.py
```
Expected: 18-day growth table ending ~146g biomass, ~6% mortality.

### Step 2 — Train the PPO Agent
```bash
# Quick training (100k steps, ~3 min CPU)
python train.py --timesteps 100000

# Full training (500k steps, ~15 min CPU / ~3 min Colab GPU)
python train.py --timesteps 500000 --seed 42
```
> A pre-trained model (`models/ppo_bsf_500000.zip`) is included — skip this step if you just want to evaluate.

### Step 3 — Evaluate All Strategies
```bash
python evaluate.py --model models/ppo_bsf_500000 --episodes 50
```
Outputs: comparison table in terminal + CSV files in `results/`

### Step 4 — Generate Advanced Analysis Plots
```bash
python advanced_analysis.py
```
Outputs: 4 behavioral analysis plots saved to `results/plots/`

### Step 5 — Run the Dashboard
```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501` — configure strategies in sidebar, click **▶ RUN SIMULATION**.

---

## 10. Project Structure

```
📂 bsf-feed-optimization/
│
├── 📁 simulator/
│   ├── __init__.py
│   └── bsf_simulator.py        ← 8D digital twin, dynamic optima, env drift
│
├── 📁 env/
│   ├── __init__.py
│   └── bsf_env.py              ← Gymnasium wrapper, 9-action discrete space
│
├── 📁 models/
│   └── ppo_bsf_500000.zip      ← Pre-trained PPO model (500k timesteps)
│
├── 📁 results/
│   ├── evaluation_results.csv  ← Per-episode results from evaluate.py
│   ├── strategy_comparison.csv ← Summary table
│   └── 📁 plots/               ← PNG plots from advanced_analysis.py
│
├── 📁 tests/
│   └── test_simulator.py       ← Manual 18-day cycle sanity check
│
├── train.py                    ← PPO training script (SB3)
├── evaluate.py                 ← Benchmarks 4 strategies head-to-head
├── advanced_analysis.py        ← Behavioral heatmaps + efficiency analysis
├── dashboard.py                ← Streamlit visual demo
└── requirements.txt            ← All Python dependencies
```

---

## 11. Future Work

### Near-Term (1–3 months)
- **Longer training** — Scale to 2M+ timesteps with Optuna hyperparameter tuning to further widen the PPO vs Rule-Based gap
- **Reward shaping** — Add explicit reward terms for C:N proximity to dynamic optimum to accelerate learning

### Medium-Term (3–6 months)
- **Weather API Integration** — Replace random environmental noise with real meteorological API data (OpenWeatherMap) for region-specific and season-specific training
- **Economic reward layer** — Add real-world cost constraints: penalize feed expenditure, electricity, labor; reward optimal harvest timing based on biomass-to-cost ratio

### Long-Term (Production)
- **Hardware deployment** — Export trained PPO model to ONNX format for deployment on Raspberry Pi 4 as an edge AI controller inside a physical BSF facility
- **Multi-batch management** — Extend the agent to manage multiple larval bins simultaneously with shared resource constraints
- **Sim-to-real transfer** — Calibrate the simulator against real sensor data from an actual farm to close the simulation gap

---

## 📚 References & Related Work

- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms* — Original PPO paper
- Čičková et al. (2015) — *The use of fly larvae for organic waste treatment* — BSF biology basis
- Gold et al. (2018) — *The Housefly and the Black Soldier Fly as Waste Managers* — Feed composition research
- Similar RL applications: Greenhouse climate control (PPO), Aquaculture feed optimization (DQN), Crop irrigation management (A3C)

---

<p align="center">
  Built with 🪲 by Anton Gilchrist A, Dev Arjun G, Jabin Joseph M<br>
  Loyola-ICAM College of Engineering and Technology · PDL 2025
</p>
