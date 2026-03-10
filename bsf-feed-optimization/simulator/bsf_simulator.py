import numpy as np

class BSFSimulator:
    """
    Simulates Black Soldier Fly (BSF) larval growth over a 18-day cycle.
    
    State Variables:
        - larval_age     : current day of the growth cycle (0 to 18)
        - biomass        : current total biomass in grams
        - cn_ratio       : Carbon-to-Nitrogen ratio of feed (10 to 40)
        - moisture       : substrate moisture percentage (30 to 90)
        - mortality_rate : cumulative % of larvae that have died
        - waste          : unconsumed feed waste in grams
        - temperature    : ambient temperature in Celsius (15 to 40)
        - humidity       : ambient relative humidity % (30 to 100)
    
    Actions (taken each day by the RL agent or manually):
        - delta_cn       : adjust C:N ratio by -2, 0, or +2
        - delta_moisture : adjust moisture by -5, 0, or +5
    """

    # ── Biological constants from research ──────────────────────────
    # Note: Optima are now dynamic properties based on larval age
    MAX_GROWTH_RATE   = 0.35   # max specific growth rate (per day)
    CYCLE_LENGTH      = 18     # days until harvest
    INITIAL_BIOMASS   = 10.0    # grams at day 0 (fresh hatch batch)

    def __init__(self, noise=True):
        """
        noise : if True, adds biological randomness to simulate
                real-world unpredictability
        """
        self.noise = noise
        self.reset()

    def reset(self):
        """Reset environment to day 0. Call this at the start of each episode."""
        self.larval_age     = 0
        self.biomass        = self.INITIAL_BIOMASS
        
        # Introduce variable starting conditions to test robustness
        if self.noise:
            self.cn_ratio       = np.random.uniform(15.0, 25.0)
            self.moisture       = np.random.uniform(55.0, 75.0)
            self.temperature    = np.random.uniform(23.0, 31.0)
            self.humidity       = np.random.uniform(60.0, 80.0)
        else:
            self.cn_ratio       = 20.0   # baseline starting C:N
            self.moisture       = 65.0   # baseline starting moisture %
            self.temperature    = 27.0   # baseline starting temperature
            self.humidity       = 70.0   # baseline starting humidity %
            
        self.mortality_rate = 0.0    # %
        self.waste          = 0.0    # grams
        self.done           = False
        return self._get_state()

    @property
    def optimal_cn(self):
        """Dynamic optimal C:N ratio: starts at 22 (high carbon/energy), drops to 14 (high protein)."""
        return 22.0 - (self.larval_age / self.CYCLE_LENGTH) * 8.0

    @property
    def optimal_moisture(self):
        """Dynamic optimal moisture: needs 75% early (wet), drops to 60% later (drier for prepupation)."""
        return 75.0 - (self.larval_age / self.CYCLE_LENGTH) * 15.0

    # ── Core growth logic ────────────────────────────────────────────
    def _cn_factor(self):
        """
        How much does the current C:N ratio help or hurt growth?
        Returns a multiplier between 0.0 and 1.0.
        """
        distance = abs(self.cn_ratio - self.optimal_cn)
        return max(0.0, 1.0 - (distance / 20.0))

    def _moisture_factor(self):
        """
        How much does current moisture help or hurt growth?
        Returns a multiplier between 0.0 and 1.0.
        """
        distance = abs(self.moisture - self.optimal_moisture)
        return max(0.0, 1.0 - (distance / 35.0))

    def _temperature_factor(self):
        """
        Optimal temperature is around 27°C.
        """
        distance = abs(self.temperature - 27.0)
        return max(0.0, 1.0 - (distance / 15.0))

    def _humidity_factor(self):
        """
        Optimal ambient humidity is around 70%.
        """
        distance = abs(self.humidity - 70.0)
        return max(0.0, 1.0 - (distance / 30.0))

    def _age_factor(self):
        """
        Larvae grow fastest between days 5–12.
        Growth slows significantly after day 14 (prepupa stage).
        """
        if self.larval_age > 14:
            return 0.1   # near-zero growth in prepupa stage
        return np.exp(-0.5 * ((self.larval_age - 8) / 4.0) ** 2)


    def _mortality_penalty(self):
        """
        Bad conditions increase mortality.
        Returns daily mortality rate as a fraction (0.0 to 1.0).
        """
        cn_stress       = max(0.0, 1.0 - self._cn_factor())
        moisture_stress = max(0.0, 1.0 - self._moisture_factor())
        temp_stress     = max(0.0, 1.0 - self._temperature_factor())
        hum_stress      = max(0.0, 1.0 - self._humidity_factor())
        
        daily_mortality = 0.002 + 0.01 * cn_stress + 0.02 * moisture_stress + 0.015 * temp_stress + 0.005 * hum_stress
        return min(daily_mortality, 0.10)   # cap at 10% per day

    # ── Step: advance one day ────────────────────────────────────────
    def step(self, delta_cn=0, delta_moisture=0):
        """
        Advance the simulation by one day.
        
        Parameters:
            delta_cn       : change in C:N ratio  (-2, 0, or +2)
            delta_moisture : change in moisture %  (-5, 0, or +5)
        
        Returns:
            state  : current state as a numpy array
            reward : reward signal for the RL agent
            done   : True if cycle is complete
            info   : dictionary with detailed metrics
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # 0. Apply natural environmental drift
        if self.noise:
            # Substrate drift
            self.moisture -= np.random.uniform(0.5, 2.5)  # daily evaporation
            self.cn_ratio += np.random.uniform(-0.5, 1.0) # feed degradation shift
            
            # Ambient weather drift
            self.temperature += np.random.uniform(-1.5, 1.5)
            self.humidity    += np.random.uniform(-3.0, 3.0)
            
            self.temperature = np.clip(self.temperature, 15.0, 40.0)
            self.humidity    = np.clip(self.humidity, 30.0, 100.0)

        # 1. Apply actions (with boundary clamping)
        self.cn_ratio  = np.clip(self.cn_ratio  + delta_cn,       10.0, 40.0)
        self.moisture  = np.clip(self.moisture  + delta_moisture, 30.0, 90.0)

        # 2. Calculate growth factors
        cn_f       = self._cn_factor()
        moist_f    = self._moisture_factor()
        temp_f     = self._temperature_factor()
        hum_f      = self._humidity_factor()
        age_f      = self._age_factor()

        # 3. Calculate daily growth rate (all factors combined)
        daily_growth_rate = self.MAX_GROWTH_RATE * cn_f * moist_f * temp_f * hum_f * age_f

        # Add biological noise (±10% randomness)
        if self.noise:
            daily_growth_rate *= np.random.uniform(0.90, 1.10)

        # 4. Update biomass
        biomass_gain   = self.biomass * daily_growth_rate
        daily_mortality = self._mortality_penalty()
        biomass_lost   = self.biomass * daily_mortality

        self.biomass        = max(0.0, self.biomass + biomass_gain - biomass_lost)
        self.mortality_rate += daily_mortality * 100  # store as %

        # 5. Calculate waste (unconsumed feed)
        feed_utilization = cn_f * moist_f * temp_f
        self.waste      += (1.0 - feed_utilization) * 2.0  # grams per day wasted

        # 6. Advance age
        self.larval_age += 1

        # 7. Calculate reward
        reward = (
            + biomass_gain * 2.0          # reward for growing
            - biomass_lost * 3.0          # heavy penalty for deaths
            - self.waste   * 0.5          # penalty for waste
        )

        # 8. Check if cycle is complete
        if self.larval_age >= self.CYCLE_LENGTH:
            self.done = True
            reward += self.biomass * 5.0

        # 9. Build info dict for logging/visualization
        info = {
            "day"           : self.larval_age,
            "biomass_g"     : round(self.biomass, 3),
            "biomass_gain_g": round(biomass_gain, 3),
            "cn_ratio"      : round(self.cn_ratio, 1),
            "moisture_pct"  : round(self.moisture, 1),
            "temperature_c" : round(self.temperature, 1),
            "humidity_pct"  : round(self.humidity, 1),
            "mortality_pct" : round(daily_mortality * 100, 2),
            "waste_g"       : round(self.waste, 3),
            "reward"        : round(reward, 3),
        }

        return self._get_state(), reward, self.done, info

    # ── State representation ─────────────────────────────────────────
    def _get_state(self):
        """
        Returns the current state as a numpy array.
        This is what the RL agent 'sees' at each step.
        
        [larval_age, biomass, cn_ratio, moisture, mortality_rate, waste, temperature, humidity]
        """
        return np.array([
            self.larval_age / self.CYCLE_LENGTH,   # normalized 0–1
            self.biomass    / 500.0,               # normalized (max ~500g)
            self.cn_ratio   / 40.0,                # normalized 0–1
            self.moisture   / 100.0,               # normalized 0–1
            self.mortality_rate / 100.0,           # normalized 0–1
            self.waste      / 50.0,                # normalized 0–1
            self.temperature / 50.0,               # normalized max 50C
            self.humidity    / 100.0,              # normalized 0-1
        ], dtype=np.float32)

    def render(self):
        """Print a simple human-readable status."""
        print(f"Day {self.larval_age:>2} | "
              f"Biomass: {self.biomass:>7.2f}g | "
              f"C:N: {self.cn_ratio:>5.1f} | "
              f"Moisture: {self.moisture:>5.1f}% | "
              f"Wait: {self.waste:>6.2f}g | "
              f"Temp: {self.temperature:>4.1f}C | "
              f"Hum: {self.humidity:>4.1f}%")