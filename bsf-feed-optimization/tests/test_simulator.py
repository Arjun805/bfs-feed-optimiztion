from simulator import BSFSimulator

def run_manual_test():
    """
    Run one full 18-day cycle manually with fixed actions.
    Use this to verify the simulator is working before RL training.
    """
    sim = BSFSimulator(noise=True)
    state = sim.reset()
    
    print("=" * 65)
    print("BSF LARVAL GROWTH SIMULATOR — MANUAL TEST RUN")
    print("=" * 65)
    sim.render()

    total_reward = 0

    for day in range(18):
        # Manual strategy: keep C:N low, moisture optimal
        # Later the RL agent will figure this out automatically
        delta_cn       = -2 if sim.cn_ratio > 18 else 0
        delta_moisture =  5 if sim.moisture < 65 else 0

        state, reward, done, info = sim.step(delta_cn, delta_moisture)
        total_reward += reward
        sim.render()

        if done:
            break

    print("=" * 65)
    print(f"HARVEST DAY | Final Biomass : {sim.biomass:.2f}g")
    print(f"             Total Reward   : {total_reward:.2f}")
    print(f"             Total Waste    : {sim.waste:.2f}g")
    print(f"             Total Mortality: {sim.mortality_rate:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    run_manual_test()