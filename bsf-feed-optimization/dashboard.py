"""
BSF Feed Optimization — Streamlit Dashboard
============================================
Phase 5: Visual Demo Dashboard

Run with:
    streamlit run dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
import uuid

# ── Add project root to path ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Page config (must be first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="BSF Feed Optimizer",
    page_icon="🪲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark green agri-tech aesthetic ──────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1a0f;
    color: #e8f5e8;
  }

  .main { background-color: #0f1a0f; }

  h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #7ddb7d;
    letter-spacing: -0.02em;
  }

  .metric-card {
    background: linear-gradient(135deg, #1a2e1a 0%, #142014 100%);
    border: 1px solid #2d5a2d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
  }

  .metric-card:hover { border-color: #7ddb7d; }

  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #7ddb7d;
    line-height: 1;
  }

  .metric-label {
    font-size: 0.75rem;
    color: #6b9b6b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
  }

  .metric-delta {
    font-size: 0.85rem;
    margin-top: 4px;
  }

  .strategy-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
  }

  .badge-ppo      { background: #1a4d1a; color: #7ddb7d; border: 1px solid #7ddb7d; }
  .badge-rule     { background: #1a3a4d; color: #7db8db; border: 1px solid #7db8db; }
  .badge-random   { background: #4d3a1a; color: #dba87d; border: 1px solid #dba87d; }
  .badge-nothing  { background: #3a1a1a; color: #db7d7d; border: 1px solid #db7d7d; }

  .stButton > button {
    background: linear-gradient(135deg, #2d5a2d, #1a3d1a);
    color: #7ddb7d;
    border: 1px solid #7ddb7d;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 10px 28px;
    transition: all 0.2s;
  }

  .stButton > button:hover {
    background: #7ddb7d;
    color: #0f1a0f;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(125, 219, 125, 0.3);
  }

  .stSelectbox label, .stSlider label {
    color: #6b9b6b;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .stSidebar { background-color: #0a130a; }

  .section-header {
    font-family: 'Space Mono', monospace;
    color: #7ddb7d;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-bottom: 1px solid #2d5a2d;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }

  .info-box {
    background: #142014;
    border-left: 3px solid #7ddb7d;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #9bc99b;
  }

  div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    color: #7ddb7d;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Import project modules
# ═══════════════════════════════════════════════════════════════════
try:
    from simulator.bsf_simulator import BSFSimulator
    from env.bsf_env import BSFEnv, ACTION_MAP
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    IMPORT_ERROR = str(e)

# ═══════════════════════════════════════════════════════════════════
# Matplotlib style
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "#0f1a0f",
    "axes.facecolor":    "#142014",
    "axes.edgecolor":    "#2d5a2d",
    "axes.labelcolor":   "#9bc99b",
    "xtick.color":       "#6b9b6b",
    "ytick.color":       "#6b9b6b",
    "text.color":        "#e8f5e8",
    "grid.color":        "#1e3a1e",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "axes.titlecolor":   "#7ddb7d",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})

COLORS = {
    "PPO Agent":   "#7ddb7d",
    "Rule-Based":  "#7db8db",
    "Random":      "#dba87d",
    "Do-Nothing":  "#db7d7d",
}


# ═══════════════════════════════════════════════════════════════════
# Simulation helpers
# ═══════════════════════════════════════════════════════════════════

def run_policy(policy_fn, n_episodes=15, noise=True):
    """Run n_episodes and return list of episode dicts + day-level traces."""
    env = BSFEnv(noise=noise)
    episodes = []
    traces   = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_data = {"days": [], "biomass": [], "cn": [], "moisture": [],
                   "waste": [], "mortality": [], "reward_cumulative": [],
                   "temperature": [], "humidity": []}
        cum_reward = 0.0
        done = False

        ep_data["days"].append(0)
        ep_data["biomass"].append(env.sim.biomass)
        ep_data["cn"].append(env.sim.cn_ratio)
        ep_data["moisture"].append(env.sim.moisture)
        ep_data["waste"].append(env.sim.waste)
        ep_data["mortality"].append(env.sim.mortality_rate)
        ep_data["temperature"].append(env.sim.temperature)
        ep_data["humidity"].append(env.sim.humidity)
        ep_data["reward_cumulative"].append(0.0)

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cum_reward += reward
            done = terminated or truncated

            ep_data["days"].append(env.sim.larval_age)
            ep_data["biomass"].append(env.sim.biomass)
            ep_data["cn"].append(env.sim.cn_ratio)
            ep_data["moisture"].append(env.sim.moisture)
            ep_data["waste"].append(env.sim.waste)
            ep_data["mortality"].append(env.sim.mortality_rate)
            ep_data["temperature"].append(env.sim.temperature)
            ep_data["humidity"].append(env.sim.humidity)
            ep_data["reward_cumulative"].append(cum_reward)

        episodes.append({
            "final_biomass": env.sim.biomass,
            "total_reward":  cum_reward,
            "total_waste":   env.sim.waste,
            "mortality":     env.sim.mortality_rate,
        })
        traces.append(ep_data)

    env.close()
    return episodes, traces


def do_nothing_policy(obs):
    return 4

def random_policy(obs):
    return np.random.randint(0, 9)

def rule_based_policy(obs):
    cn_ratio = obs[2] * 40.0
    moisture = obs[3] * 100.0
    cn_action    = 0 if cn_ratio > 19.0 else (2 if cn_ratio < 16.0 else 1)
    moist_action = 0 if moisture > 72.0 else (2 if moisture < 63.0 else 1)
    return cn_action * 3 + moist_action

def get_ppo_policy(model_path):
    """Load PPO model and return a policy function."""
    try:
        from stable_baselines3 import PPO
        full_path = model_path
        if not full_path.endswith(".zip"):
            full_path += ".zip"
        # Handle relative paths
        if not os.path.isabs(full_path):
            full_path = os.path.join(PROJECT_ROOT, full_path)
        if not os.path.exists(full_path):
            return None, False, f"Model file not found: {full_path}"
        model = PPO.load(model_path if os.path.isabs(model_path) else os.path.join(PROJECT_ROOT, model_path))
        def ppo_policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        return ppo_policy, True, ""
    except Exception as e:
        return None, False, str(e)


# ═══════════════════════════════════════════════════════════════════
# Chart helpers
# ═══════════════════════════════════════════════════════════════════

def plot_biomass_growth(strategy_traces):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, (_, traces) in strategy_traces.items():
        color = COLORS.get(name, "#ffffff")
        # Plot all episode traces faintly
        for tr in traces:
            ax.plot(tr["days"], tr["biomass"], color=color, alpha=0.12, linewidth=0.8)
        # Bold mean line
        max_len = max(len(tr["days"]) for tr in traces)
        mean_biomass = [
            np.mean([tr["biomass"][d] for tr in traces if d < len(tr["biomass"])])
            for d in range(max_len)
        ]
        ax.plot(range(max_len), mean_biomass, color=color, linewidth=2.5,
                label=name, zorder=5)

    ax.set_xlabel("Day of Growth Cycle")
    ax.set_ylabel("Biomass (g)")
    ax.set_title("Biomass Growth Over 18-Day Cycle")
    ax.legend(loc="upper left", framealpha=0.3,
              facecolor="#142014", edgecolor="#2d5a2d")
    ax.grid(True)
    ax.set_xlim(0, 18)
    fig.tight_layout()
    return fig


def plot_cn_moisture(traces_dict, strategy_name):
    """Plot C:N and moisture decisions for one strategy's last episode."""
    _, traces = traces_dict[strategy_name]
    tr = traces[-1]
    color = COLORS.get(strategy_name, "#ffffff")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    ax1.plot(tr["days"], tr["cn"], color=color, linewidth=2)
    ax1.axhline(17.5, color="#7ddb7d", linestyle="--", alpha=0.5, linewidth=1)
    ax1.fill_between(tr["days"], 15, 20, alpha=0.1, color="#7ddb7d")
    ax1.set_ylabel("C:N Ratio")
    ax1.set_title(f"Feed Decisions — {strategy_name}")
    ax1.set_ylim(8, 42)
    ax1.grid(True)
    ax1.text(17.2, 17.5, "optimal", color="#7ddb7d", fontsize=7, va="bottom", ha="right")

    ax2.plot(tr["days"], tr["moisture"], color="#a8d8db", linewidth=2)
    ax2.axhline(67.5, color="#7ddb7d", linestyle="--", alpha=0.5, linewidth=1)
    ax2.fill_between(tr["days"], 60, 75, alpha=0.1, color="#7ddb7d")
    ax2.set_ylabel("Moisture (%)")
    ax2.set_xlabel("Day")
    ax2.set_ylim(28, 92)
    ax2.grid(True)
    ax2.text(17.2, 67.5, "optimal", color="#7ddb7d", fontsize=7, va="bottom", ha="right")

    fig.tight_layout()
    return fig


def plot_comparison_bars(summary_df):
    strategies = summary_df.index.tolist()
    colors = [COLORS.get(s, "#888") for s in strategies]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("avg_biomass",   "Avg Final Biomass (g)", "Higher = Better"),
        ("avg_waste",     "Avg Total Waste (g)",   "Lower = Better"),
        ("avg_mortality", "Avg Mortality (%)",      "Lower = Better"),
    ]

    for ax, (col, label, note) in zip(axes, metrics):
        vals = summary_df[col].values
        bars = ax.barh(strategies, vals, color=colors, alpha=0.85,
                       edgecolor="#2d5a2d", linewidth=0.5, height=0.55)
        ax.set_xlabel(label)
        ax.set_title(note, fontsize=8, color="#6b9b6b")
        ax.grid(True, axis="x")
        for bar, val in zip(bars, vals):
            ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8.5, color="#e8f5e8")
        ax.set_xlim(0, max(vals) * 1.18)

    fig.tight_layout(pad=2)
    return fig


def plot_reward_episodes(episodes_list):
    """Plot total reward per episode for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for name, (episodes, _) in episodes_list.items():
        color = COLORS.get(name, "#ffffff")
        rewards = [ep["total_reward"] for ep in episodes]
        ax.plot(range(1, len(rewards) + 1), rewards,
                color=color, linewidth=2, label=name, marker="o",
                markersize=3, alpha=0.9)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward by Strategy")
    ax.legend(framealpha=0.3, facecolor="#142014", edgecolor="#2d5a2d")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_3d_state_space(traces_dict, strategy_name):
    _, traces = traces_dict[strategy_name]
    tr = traces[-1]  # Take the last episode

    fig = go.Figure(data=[go.Scatter3d(
        x=tr["temperature"],
        y=tr["humidity"],
        z=tr["biomass"],
        mode='lines+markers',
        marker=dict(
            size=5,
            color=tr["days"],
            colorscale='Viridis',
            colorbar=dict(title='Day', thickness=15),
            opacity=0.8
        ),
        line=dict(
            color='#7ddb7d',
            width=2
        ),
        text=[f"Day: {d}<br>C:N: {c:.1f}<br>Moist: {m:.1f}%" for d, c, m in zip(tr["days"], tr["cn"], tr["moisture"])],
        hoverinfo='text+x+y+z'
    )])

    fig.update_layout(
        title=f"3D State Trajectory — {strategy_name}",
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Humidity (%)',
            zaxis_title='Biomass (g)',
            xaxis=dict(gridcolor='#2d5a2d', backgroundcolor='#142014', color='#6b9b6b'),
            yaxis=dict(gridcolor='#2d5a2d', backgroundcolor='#142014', color='#6b9b6b'),
            zaxis=dict(gridcolor='#2d5a2d', backgroundcolor='#142014', color='#6b9b6b'),
        ),
        margin=dict(r=20, b=10, l=10, t=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f5e8')
    )
    return fig


def generate_pdf_report(summary_df, all_results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(200, 10, txt="BSF Feed Optimization - Executive Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt="Strategy Performance Summary:", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", style="B", size=9)
    col_widths = [45, 30, 30, 30, 30]
    headers = ["Strategy", "Avg Biomass(g)", "Avg Reward", "Avg Waste(g)", "Avg Mortality(%)"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, h, border=1, align='C')
    pdf.ln()

    pdf.set_font("Helvetica", size=9)
    for idx, row in summary_df.iterrows():
        pdf.cell(col_widths[0], 10, str(idx), border=1)
        pdf.cell(col_widths[1], 10, f"{row['avg_biomass']:.1f}", border=1, align='C')
        pdf.cell(col_widths[2], 10, f"{row['avg_reward']:.1f}", border=1, align='C')
        pdf.cell(col_widths[3], 10, f"{row['avg_waste']:.1f}", border=1, align='C')
        pdf.cell(col_widths[4], 10, f"{row['avg_mortality']:.1f}", border=1, align='C')
        pdf.ln()

    pdf.ln(10)

    # Add bar chart
    fig1 = plot_comparison_bars(summary_df)
    # patch colors for white background pdf printing
    fig1.patch.set_facecolor('white')
    for ax in fig1.axes:
        ax.set_facecolor('white')
        ax.xaxis.label.set_color('black')
        ax.title.set_color('black')
        ax.tick_params(colors='black')
        for txt in ax.texts:
            txt.set_color('black')
        
    img_path1 = os.path.join(tempfile.gettempdir(), f"chart1_{uuid.uuid4().hex[:8]}.png")
    fig1.savefig(img_path1, bbox_inches="tight", facecolor="white")
    pdf.image(img_path1, x=10, w=190)
    plt.close(fig1)

    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(200, 10, txt="Biomass Growth Over 18-Day Cycle", ln=True)
    
    fig2 = plot_biomass_growth(all_results)
    fig2.patch.set_facecolor('white')
    for ax in fig2.axes:
        ax.set_facecolor('white')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        ax.tick_params(colors='black')
        if ax.get_legend():
            for text in ax.get_legend().get_texts():
                text.set_color('black')

    img_path2 = os.path.join(tempfile.gettempdir(), f"chart2_{uuid.uuid4().hex[:8]}.png")
    fig2.savefig(img_path2, bbox_inches="tight", facecolor="white")
    pdf.image(img_path2, x=10, w=190)
    plt.close(fig2)

    pdf_path = os.path.join(tempfile.gettempdir(), f"report_{uuid.uuid4().hex[:8]}.pdf")
    pdf.output(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # cleanup temp images
    try: os.remove(img_path1)
    except: pass
    try: os.remove(img_path2)
    except: pass
    try: os.remove(pdf_path)
    except: pass
    
    return pdf_bytes


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🪲 BSF Optimizer")
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    default_model = os.path.join(PROJECT_ROOT, "models", "ppo_bsf_500000")
    model_path = st.text_input(
        "Trained Model Path",
        value=default_model,
        help="Absolute path to your .zip model (without extension)"
    )

    n_episodes = st.slider("Episodes per strategy", 5, 30, 15)
    use_noise  = st.toggle("Biological noise", value=True)

    st.markdown('<div class="section-header">Strategies</div>', unsafe_allow_html=True)
    show_ppo     = st.checkbox("🟢 PPO Agent",   value=True)
    show_rule    = st.checkbox("🔵 Rule-Based",  value=True)
    show_random  = st.checkbox("🟠 Random",      value=True)
    show_nothing = st.checkbox("🔴 Do-Nothing",  value=True)

    run_btn = st.button("▶  RUN SIMULATION", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4d7a4d; line-height:1.6'>
    <b>Phase 7 — 3D & Executive Reporting</b><br><br>
    Added interactive <b>3D State Trajectory</b> visualization and <b>Automated PDF Performance Audits</b>.<br><br>
    <b>State (8D):</b> age, biomass, C:N, moist, mort, waste, temp, hum<br>
    <b>Actions:</b> 9 discrete (Δ C:N × Δ moisture)<br>
    <b>Reward:</b> biomass gain − mortality − waste
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='margin-bottom:0'>AI-Driven Feed Optimization</h1>
<p style='color:#6b9b6b; font-size:0.9rem; margin-top:4px; font-family:monospace'>
  Black Soldier Fly Larvae · Reinforcement Learning · PPO Agent
</p>
""", unsafe_allow_html=True)

st.markdown("---")

if not MODULES_LOADED:
    st.error(f"⚠️ Could not import project modules: {IMPORT_ERROR}")
    st.info("Make sure `simulator/` and `env/` folders are in the same directory as dashboard.py")
    st.stop()

# ═══════════════════════════════════════════════════════════════════
# Run on button press
# ═══════════════════════════════════════════════════════════════════

if run_btn:
    strategies_to_run = {}
    if show_nothing:
        strategies_to_run["Do-Nothing"] = do_nothing_policy
    if show_random:
        strategies_to_run["Random"] = random_policy
    if show_rule:
        strategies_to_run["Rule-Based"] = rule_based_policy
    if show_ppo:
        ppo_fn, loaded, err_msg = get_ppo_policy(model_path)
        if loaded:
            strategies_to_run["PPO Agent"] = ppo_fn
        else:
            st.warning(f"⚠️ Could not load PPO model: {err_msg}")

    if not strategies_to_run:
        st.warning("Select at least one strategy.")
        st.stop()

    # Run all strategies
    progress = st.progress(0, text="Running simulations...")
    all_results = {}
    for i, (name, fn) in enumerate(strategies_to_run.items()):
        progress.progress((i + 1) / len(strategies_to_run),
                          text=f"Simulating {name}...")
        episodes, traces = run_policy(fn, n_episodes=n_episodes, noise=use_noise)
        all_results[name] = (episodes, traces)
    progress.empty()

    # Build summary dataframe
    rows = []
    for name, (episodes, _) in all_results.items():
        rows.append({
            "Strategy":     name,
            "avg_biomass":  round(np.mean([e["final_biomass"] for e in episodes]), 1),
            "best_biomass": round(np.max([e["final_biomass"] for e in episodes]), 1),
            "avg_reward":   round(np.mean([e["total_reward"]  for e in episodes]), 1),
            "avg_waste":    round(np.mean([e["total_waste"]   for e in episodes]), 1),
            "avg_mortality": round(np.mean([e["mortality"]    for e in episodes]), 1),
        })
    summary_df = pd.DataFrame(rows).set_index("Strategy")
    summary_df = summary_df.sort_values("avg_biomass", ascending=False)

    # ── KPI cards ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Key Results</div>', unsafe_allow_html=True)

    best_strategy = summary_df.index[0]
    worst_strategy = summary_df.index[-1]
    best_biomass = summary_df.loc[best_strategy, "avg_biomass"]
    worst_biomass = summary_df.loc[worst_strategy, "avg_biomass"]
    improvement = ((best_biomass - worst_biomass) / max(worst_biomass, 0.1)) * 100

    cols = st.columns(4)
    kpis = [
        ("🏆 Best Strategy", best_strategy, "by avg biomass"),
        ("📈 Best Biomass",  f"{best_biomass}g", f"avg over {n_episodes} episodes"),
        ("📉 Worst Biomass", f"{worst_biomass}g", f"{worst_strategy}"),
        ("⚡ Improvement",  f"+{improvement:.0f}%", "best vs worst"),
    ]
    for col, (label, val, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
              <div class="metric-delta" style="color:#6b9b6b">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary table ────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Strategy Comparison Table</div>', unsafe_allow_html=True)

    display_df = summary_df.rename(columns={
        "avg_biomass":  "Avg Biomass (g)",
        "best_biomass": "Best Biomass (g)",
        "avg_reward":   "Avg Reward",
        "avg_waste":    "Avg Waste (g)",
        "avg_mortality": "Avg Mortality (%)",
    })
    st.dataframe(
        display_df.style
        .background_gradient(subset=["Avg Biomass (g)", "Best Biomass (g)"], cmap="Greens")
        .background_gradient(subset=["Avg Waste (g)", "Avg Mortality (%)"], cmap="Reds_r")
        .format("{:.1f}"),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Biomass Growth Curves</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Each bold line = <b>mean biomass</b> across episodes. Faint lines = individual episodes (biological noise variability).</div>', unsafe_allow_html=True)
    st.pyplot(plot_biomass_growth(all_results))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison bars ──────────────────────────────────────────
    st.markdown('<div class="section-header">🏁 Head-to-Head Comparison</div>', unsafe_allow_html=True)
    st.pyplot(plot_comparison_bars(summary_df))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Reward across episodes ───────────────────────────────────
    st.markdown('<div class="section-header">🎯 Episode Rewards</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Total reward per episode. A well-trained PPO agent should show <b>higher, more consistent rewards</b>.</div>', unsafe_allow_html=True)
    st.pyplot(plot_reward_episodes(all_results))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feed decision inspector ──────────────────────────────────
    st.markdown('<div class="section-header">🔬 Feed Decision Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">How each strategy adjusts C:N ratio and moisture over the cycle. Green zone = optimal biological range.</div>', unsafe_allow_html=True)

    selected = st.selectbox("Select strategy to inspect:", list(all_results.keys()))
    st.pyplot(plot_cn_moisture(all_results, selected))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3D State Map ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🌌 3D State Space Trajectory</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Explore how the agent navigates Temperature, Humidity, and Biomass interactively. Green line tracks the path across the 18-day cycle.</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_3d_state_space(all_results, selected), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Raw data & PDF download ──────────────────────────────────
    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)

    all_rows = []
    for name, (episodes, _) in all_results.items():
        for i, ep in enumerate(episodes):
            all_rows.append({"strategy": name, "episode": i + 1, **ep})
    export_df = pd.DataFrame(all_rows)

    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            label="⬇  Download CSV Data",
            data=export_df.to_csv(index=False).encode(),
            file_name="bsf_evaluation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with colB:
        with st.spinner("Generating PDF Report..."):
            pdf_bytes = generate_pdf_report(summary_df, all_results)
        st.download_button(
            label="📄  Download Executive PDF Report",
            data=pdf_bytes,
            file_name="bsf_executive_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

else:
    # ── Landing state ────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px'>
      <div style='font-size:5rem'>🪲</div>
      <h2 style='color:#7ddb7d; margin-top:20px'>Ready to Optimize</h2>
      <p style='color:#6b9b6b; max-width:500px; margin:0 auto; line-height:1.7'>
        Configure your strategies in the sidebar and click
        <b style='color:#7ddb7d'>▶ RUN SIMULATION</b> to compare how the
        PPO agent performs against rule-based and random baselines
        across an 18-day BSF larval growth cycle.
      </p>
      <br>
      <div style='display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:20px'>
        <span class="strategy-badge badge-ppo">🟢 PPO Agent</span>
        <span class="strategy-badge badge-rule">🔵 Rule-Based</span>
        <span class="strategy-badge badge-random">🟠 Random</span>
        <span class="strategy-badge badge-nothing">🔴 Do-Nothing</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🧬 The Simulator**

        A mathematical model of BSF larval biology. C:N ratio and moisture
        drive daily growth. Age determines growth phase. Noise adds realism.
        """)
    with col2:
        st.markdown("""
        **🤖 The RL Agent (PPO)**

        Trained over 100,000+ timesteps to find the optimal daily feeding
        strategy. Maximizes biomass while minimizing waste and mortality.
        """)
    with col3:
        st.markdown("""
        **📊 The Comparison**

        4 strategies compete across multiple 18-day episodes. Results show
        biomass, waste, mortality, and reward — proving RL outperforms rules.
        """)