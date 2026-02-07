"""
F1 Pit Stop Strategy Simulator (MVP)

What this script does:
1) Loads real F1 lap data (FastF1)
2) Cleans laps to remove pit laps and extreme outliers
3) Builds stint-relative lap index (stint_lap = tire age)
4) Fits tire degradation per stint (slope of lap_time vs stint_lap)
5) Aggregates compound-level parameters (median slope + intercept)
6) Runs a counterfactual 1-stop strategy simulation to find optimal pit lap
7) Plots pit lap vs simulated total race time and saves a PNG

Notes / assumptions (important for readers):
- This is a simplified model: no traffic, no safety cars, no variable pit loss
- "Counterfactual" means we simulate lap times from the fitted model rather than
  using actual lap times after changing the pit lap.
"""

import fastf1 as ff1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_race(year: int, gp_name: str):
    """
    Create a FastF1 session and load race data.

    FastF1 uses a cache directory to avoid redownloading data every run.
    """
    ff1.Cache.enable_cache("cache")
    session = ff1.get_session(year, gp_name, "R")  # "R" = Race session
    session.load()
    return session


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw lap data for modeling.

    Steps:
    - Drop laps without LapTime
    - Remove in/out pit laps (unrepresentative pace)
    - Convert LapTime to seconds
    - Remove extreme outliers per driver (5th to 95th percentile)

    Returns a smaller dataframe with only columns we need for modeling.
    """
    df = laps.copy()

    # Keep only laps with a recorded lap time
    df = df[df["LapTime"].notna()]

    # Remove in-laps/out-laps (not representative "normal pace")
    if "PitInLap" in df.columns:
        df = df[~df["PitInLap"].fillna(False)]
    if "PitOutLap" in df.columns:
        df = df[~df["PitOutLap"].fillna(False)]

    # Convert LapTime (timedelta) -> numeric seconds for modeling
    df = df.assign(lap_time_s=df["LapTime"].dt.total_seconds())

    # Drop extreme slow/fast laps per driver (reduces yellow flags/incidents/noise)
    def clip_driver(g: pd.DataFrame) -> pd.DataFrame:
        lo = g["lap_time_s"].quantile(0.05)
        hi = g["lap_time_s"].quantile(0.95)
        return g[(g["lap_time_s"] >= lo) & (g["lap_time_s"] <= hi)]

    df = df.groupby("Driver", group_keys=False).apply(clip_driver)

    keep = ["Driver", "LapNumber", "Stint", "Compound", "lap_time_s"]
    return df[keep].sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def add_stint_lap_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'stint_lap' which counts laps within each (Driver, Stint).

    This is our proxy for tire age.
    """
    out = df.copy()
    out["stint_lap"] = out.groupby(["Driver", "Stint"]).cumcount() + 1
    return out


def fit_degradation_per_stint(df: pd.DataFrame, driver: str, min_stint_laps: int = 8):
    """
    Fit degradation (seconds lost per lap) within each stint for a given driver.

    For each (Stint, Compound), fit a line:
        lap_time_s = intercept + slope * stint_lap

    Then return:
    - median slope across stints (robust single degradation estimate)
    - a dataframe of all stint-level slopes and intercepts
    """
    d = df[df["Driver"] == driver].copy()

    # Drop the first lap of each stint (often a bit noisy even after cleaning)
    d = d[d["stint_lap"] >= 2]

    stint_slopes = []

    for (stint, compound), g in d.groupby(["Stint", "Compound"]):
        if len(g) < min_stint_laps:
            continue

        x = g["stint_lap"].values
        y = g["lap_time_s"].values

        # Fit y = m*x + b for this stint
        m, b = np.polyfit(x, y, 1)

        stint_slopes.append(
            {
                "Driver": driver,
                "Stint": stint,
                "Compound": compound,
                "n_laps": len(g),
                "slope_sec_per_lap": m,
                "intercept_sec": b,
            }
        )

    if not stint_slopes:
        print(f"No stints long enough for {driver}. Try lowering min_stint_laps.")
        return None

    slopes_df = pd.DataFrame(stint_slopes)
    median_slope = slopes_df["slope_sec_per_lap"].median()

    print(f"\nDriver: {driver}")
    print(
        slopes_df.sort_values("slope_sec_per_lap")[
            ["Stint", "Compound", "n_laps", "slope_sec_per_lap"]
        ].to_string(index=False)
    )
    print(f"\nMedian degradation (sec/lap): {median_slope:.4f}")

    return median_slope, slopes_df


def get_compound_params(slopes_df: pd.DataFrame):
    """
    Aggregate stint-level fits into compound-level parameters.

    For each Compound:
    - deg_rate = median slope across stints on that compound
    - base_pace = median intercept across stints on that compound
      (interpretable as "fresh-ish tire pace" under this linear model)

    Returns: dict[compound] -> {deg_rate, base_pace, n_stints}
    """
    params = {}
    for compound, g in slopes_df.groupby("Compound"):
        params[compound] = {
            "deg_rate": float(g["slope_sec_per_lap"].median()),
            "base_pace": float(g["intercept_sec"].median()),
            "n_stints": int(g.shape[0]),
        }
    return params


def simulate_counterfactual_one_stop(
    driver_laps: pd.DataFrame,
    pre_compound: str,
    post_compound: str,
    compound_params: dict,
    pit_loss: float = 22.0,
):
    """
    Counterfactual one-stop simulation.

    For each possible pit lap L:
    - simulate laps 1..L on pre_compound using (base_pace + deg_rate * stint_lap)
    - add pit_loss
    - simulate laps L+1..end on post_compound using its params
    - compute total simulated race time

    Returns a dataframe with (pit_lap, total_time).
    """
    total_laps = int(driver_laps["LapNumber"].max())

    if pre_compound not in compound_params or post_compound not in compound_params:
        raise ValueError(f"Missing compound params. Have: {list(compound_params.keys())}")

    pre_base = compound_params[pre_compound]["base_pace"]
    pre_deg = compound_params[pre_compound]["deg_rate"]

    post_base = compound_params[post_compound]["base_pace"]
    post_deg = compound_params[post_compound]["deg_rate"]

    results = []

    # Avoid pitting too early/late (simple guardrails)
    for pit_lap in range(10, total_laps - 10):
        # Laps 1..pit_lap on pre compound
        pre_stint_laps = np.arange(1, pit_lap + 1)
        time_pre = float((pre_base + pre_deg * pre_stint_laps).sum())

        # Laps pit_lap+1..end on post compound
        post_len = total_laps - pit_lap
        post_stint_laps = np.arange(1, post_len + 1)
        time_post = float((post_base + post_deg * post_stint_laps).sum())

        total_time = time_pre + pit_loss + time_post
        results.append((pit_lap, total_time))

    results_df = pd.DataFrame(results, columns=["pit_lap", "total_time"])

    best_row = results_df.loc[results_df["total_time"].idxmin()]
    best_lap = int(best_row["pit_lap"])

    threshold = float(best_row["total_time"]) + 3.0
    window = results_df[results_df["total_time"] <= threshold]
    w0, w1 = int(window["pit_lap"].min()), int(window["pit_lap"].max())

    print(f"\nCounterfactual strategy: {pre_compound} → {post_compound}")
    print(f"Best pit lap: {best_lap}")
    print(f"Pit window (within +3s): {w0}–{w1}")
    print(results_df.sort_values("total_time").head(10).to_string(index=False))

    return results_df


def main():
    # ------------------------------
    # User inputs (change these)
    # ------------------------------
    year = 2024
    gp_name = "Bahrain"
    driver_code = "LEC"  # e.g., "VER", "LEC", "NOR"

    print(f"\nRunning strategy for {driver_code} at {gp_name} {year}")

    # 1) Load race data
    session = load_race(year, gp_name)
    laps = session.laps

    # 2) Clean data + add tire-age index
    clean = add_stint_lap_index(clean_laps(laps))

    print("Raw laps shape:", laps.shape)
    print("Cleaned laps shape:", clean.shape)
    print("\nCompounds present:")
    print(clean["Compound"].value_counts())

    # 3) Fit degradation per stint for this driver
    fit_result = fit_degradation_per_stint(clean, driver_code, min_stint_laps=8)
    if fit_result is None:
        return

    _, slopes_df = fit_result

    # 4) Build compound params (deg rate + base pace)
    params = get_compound_params(slopes_df)

    print("\nCompound params (median):")
    for c, p in params.items():
        print(f"{c}: deg_rate={p['deg_rate']:.4f} sec/lap, base_pace={p['base_pace']:.2f} sec, stints={p['n_stints']}")

    # 5) Filter to just this driver's laps (for total_laps computation)
    driver_laps = clean[clean["Driver"] == driver_code].copy()
    if driver_laps.empty:
        print(f"No laps found for driver {driver_code}. Check the driver code.")
        return

    # 6) Run main counterfactual sim (choose a starting and target compound)
    # For Bahrain 2024, many drivers started SOFT then went HARD in the race.
    # If your driver didn't use SOFT in the cleaned data, you'll need to pick a compound that exists in params.
    pre_compound = "SOFT"
    post_compound = "HARD"

    res = simulate_counterfactual_one_stop(
        driver_laps,
        pre_compound=pre_compound,
        post_compound=post_compound,
        compound_params=params,
        pit_loss=22.0,
    )

    # 7) Plot strategy curve + mark optimum
    best_idx = res["total_time"].idxmin()
    best_lap = res.loc[best_idx, "pit_lap"]
    best_time = res.loc[best_idx, "total_time"]

    plt.figure()
    plt.plot(res["pit_lap"], res["total_time"])
    plt.scatter(best_lap, best_time)
    plt.xlabel("Pit lap")
    plt.ylabel("Simulated total race time (s)")
    plt.title(f"Pit lap vs simulated total race time ({driver_code}, {gp_name} {year})")

    plt.savefig("pit_strategy_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()