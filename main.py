import fastf1 as ff1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_race(year: int, gp_name: str):
    ff1.Cache.enable_cache("cache")
    session = ff1.get_session(year, gp_name, "R")
    session.load()
    return session


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()

    # Keep only laps with a recorded lap time
    df = df[df["LapTime"].notna()]

    # Remove in-laps/out-laps
    if "PitInLap" in df.columns:
        df = df[~df["PitInLap"].fillna(False)]
    if "PitOutLap" in df.columns:
        df = df[~df["PitOutLap"].fillna(False)]

    # Convert lap time to seconds
    df = df.assign(lap_time_s=df["LapTime"].dt.total_seconds())

    # Drop extreme outliers per driver
    def clip_driver(g: pd.DataFrame) -> pd.DataFrame:
        lo = g["lap_time_s"].quantile(0.05)
        hi = g["lap_time_s"].quantile(0.95)
        return g[(g["lap_time_s"] >= lo) & (g["lap_time_s"] <= hi)]

    df = df.groupby("Driver", group_keys=False).apply(clip_driver)

    keep = ["Driver", "LapNumber", "Stint", "Compound", "lap_time_s"]
    return df[keep].sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def add_stint_lap_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stint_lap"] = out.groupby(["Driver", "Stint"]).cumcount() + 1
    return out


def fit_degradation_per_stint(df: pd.DataFrame, driver: str, min_stint_laps: int = 8):
    d = df[df["Driver"] == driver].copy()

    # optional: drop first lap of each stint (often outlap-ish behavior even after cleaning)
    d = d[d["stint_lap"] >= 2]

    stint_slopes = []

    for (stint, compound), g in d.groupby(["Stint", "Compound"]):
        if len(g) < min_stint_laps:
            continue

        x = g["stint_lap"].values
        y = g["lap_time_s"].values

        # Fit y = m*x + b inside this single stint
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

    # Robust aggregate: median slope across stints
    median_slope = slopes_df["slope_sec_per_lap"].median()

    print(f"\nDriver: {driver}")
    print(slopes_df.sort_values("slope_sec_per_lap")[["Stint", "Compound", "n_laps", "slope_sec_per_lap"]].to_string(index=False))
    print(f"\nMedian degradation (sec/lap): {median_slope:.4f}")

    return median_slope, slopes_df



def get_compound_params(slopes_df: pd.DataFrame):
    """
    Returns per-compound (deg_rate, base_pace) using medians across stints.
    base_pace is estimated from intercepts; deg_rate from slopes.
    """
    params = {}
    for compound, g in slopes_df.groupby("Compound"):
        params[compound] = {
            "deg_rate": float(g["slope_sec_per_lap"].median()),
            "base_pace": float(g["intercept_sec"].median()),
            "n_stints": int(g.shape[0]),
        }
    return params



def simulate_pit_strategy(df: pd.DataFrame, driver: str,
                           deg_rate: float,
                           pit_loss: float = 22.0):
    d = df[df["Driver"] == driver].copy()

    total_laps = int(d["LapNumber"].max())

    # Estimate fresh-tire pace
    base_pace = d["lap_time_s"].min()

    results = []

    # try many pit laps
    for pit_lap in range(10, total_laps - 10):
        before = d[d["LapNumber"] <= pit_lap]
        time_before = before["lap_time_s"].sum()

        laps_after = total_laps - pit_lap

        # fresh tires after pit
        new_stint_laps = np.arange(1, laps_after + 1)
        predicted_after = base_pace + deg_rate * new_stint_laps

        time_after = predicted_after.sum()

        total_time = time_before + pit_loss + time_after

        results.append((pit_lap, total_time))

    results_df = pd.DataFrame(results,
                              columns=["pit_lap", "total_time"])

    best_row = results_df.loc[results_df["total_time"].idxmin()]
    best_lap = int(best_row["pit_lap"])

    print(f"\nBest pit lap for {driver}: {best_lap}")

    # define pit window = laps within 0.5 sec of optimal
    threshold = best_row["total_time"] + 3.0
    window = results_df[
        results_df["total_time"] <= threshold
    ]

    window_start = int(window["pit_lap"].min())
    window_end = int(window["pit_lap"].max())

    print(f"Recommended pit window: {window_start}–{window_end}")
    print("\nTop 10 pit lap options:")
    print(results_df.sort_values("total_time").head(10).to_string(index=False))
    return results_df


def simulate_pit_strategy_compound(df: pd.DataFrame, driver: str,
                                  compound_params: dict,
                                  pit_loss: float = 22.0,
                                  post_pit_compound: str = "HARD"):
    d = df[df["Driver"] == driver].copy()
    total_laps = int(d["LapNumber"].max())

    if post_pit_compound not in compound_params:
        raise ValueError(f"No params for compound {post_pit_compound}. Available: {list(compound_params.keys())}")

    base_pace = compound_params[post_pit_compound]["base_pace"]
    deg_rate = compound_params[post_pit_compound]["deg_rate"]

    results = []

    for pit_lap in range(10, total_laps - 10):
        before = d[d["LapNumber"] <= pit_lap]
        time_before = before["lap_time_s"].sum()

        laps_after = total_laps - pit_lap
        new_stint_laps = np.arange(1, laps_after + 1)

        predicted_after = base_pace + deg_rate * new_stint_laps
        time_after = predicted_after.sum()

        total_time = time_before + pit_loss + time_after
        results.append((pit_lap, total_time))

    results_df = pd.DataFrame(results, columns=["pit_lap", "total_time"])
    best_row = results_df.loc[results_df["total_time"].idxmin()]
    best_lap = int(best_row["pit_lap"])

    threshold = float(best_row["total_time"]) + 3.0
    window = results_df[results_df["total_time"] <= threshold]
    window_start = int(window["pit_lap"].min())
    window_end = int(window["pit_lap"].max())

    print(f"\nPost-pit compound: {post_pit_compound}")
    print(f"Best pit lap for {driver}: {best_lap}")
    print(f"Recommended pit window: {window_start}–{window_end}")

    print("\nTop 10 pit lap options:")
    print(results_df.sort_values("total_time").head(10).to_string(index=False))

    return results_df

def simulate_counterfactual_one_stop(driver_laps: pd.DataFrame,
                                     pre_compound: str,
                                     post_compound: str,
                                     compound_params: dict,
                                     pit_loss: float = 22.0):
    total_laps = int(driver_laps["LapNumber"].max())

    if pre_compound not in compound_params or post_compound not in compound_params:
        raise ValueError(f"Missing compound params. Have: {list(compound_params.keys())}")

    pre_base = compound_params[pre_compound]["base_pace"]
    pre_deg = compound_params[pre_compound]["deg_rate"]

    post_base = compound_params[post_compound]["base_pace"]
    post_deg = compound_params[post_compound]["deg_rate"]

    results = []

    for pit_lap in range(10, total_laps - 10):
        # simulate laps 1..pit_lap on pre compound
        pre_stint_laps = np.arange(1, pit_lap + 1)
        pre_times = pre_base + pre_deg * pre_stint_laps
        time_pre = float(pre_times.sum())

        # simulate laps pit_lap+1..total on post compound
        post_len = total_laps - pit_lap
        post_stint_laps = np.arange(1, post_len + 1)
        post_times = post_base + post_deg * post_stint_laps
        time_post = float(post_times.sum())

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
    year = 2024
    gp_name = "Bahrain"

    session = load_race(year, gp_name)
    laps = session.laps

    clean = add_stint_lap_index(clean_laps(laps))

    print("Raw laps shape:", laps.shape)
    print("Cleaned laps shape:", clean.shape)

    print("\nClean preview:")
    print(clean.head(12))

    print("\nCompounds present:")

    print(clean["Compound"].value_counts())

    median_deg, ham_slopes = fit_degradation_per_stint(clean, "HAM", min_stint_laps=8)

    deg_rate, _ = fit_degradation_per_stint(clean, "HAM")
    simulate_pit_strategy(clean, "HAM", deg_rate)

    median_deg, ham_slopes = fit_degradation_per_stint(clean, "HAM", min_stint_laps=8)
    params = get_compound_params(ham_slopes)

    # ---- Fit degradation once ----
    median_deg, ham_slopes = fit_degradation_per_stint(clean, "HAM", min_stint_laps=8)
    params = get_compound_params(ham_slopes)

    print("\nCompound params (median):")
    for c, p in params.items():
        print(f"{c}: deg_rate={p['deg_rate']:.4f} sec/lap, base_pace={p['base_pace']:.2f} sec, stints={p['n_stints']}")

    # ---- Define HAM dataframe BEFORE using it ----
    ham = clean[clean["Driver"] == "HAM"].copy()

    # ---- Counterfactual strategy + plot (this is the main artifact) ----
    res = simulate_counterfactual_one_stop(
        ham,
        pre_compound="SOFT",
        post_compound="HARD",
        compound_params=params,
        pit_loss=22.0
    )


    best_idx = res["total_time"].idxmin()
    best_lap = res.loc[best_idx, "pit_lap"]
    best_time = res.loc[best_idx, "total_time"]

    plt.figure()
    plt.plot(res["pit_lap"], res["total_time"])
    plt.scatter(best_lap, best_time)
    plt.xlabel("Pit lap")
    plt.ylabel("Simulated total race time (s)")
    plt.title("Pit lap vs simulated total race time (HAM, Bahrain 2024)")

    plt.savefig("pit_strategy_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Optional: compare post-pit SOFT if you want
    if "SOFT" in params:
        simulate_counterfactual_one_stop(
            ham,
            pre_compound="SOFT",
            post_compound="SOFT",
            compound_params=params,
            pit_loss=22.0
        )

if __name__ == "__main__":
    main()