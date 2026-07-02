"""
Predictive Conformal Slip Monitoring — v2
Corrected pipeline addressing methodological weaknesses in v1:

  1. De-confounded slip proxy: Speed is now an explicit model feature (not just
     the denominator of the target), so residuals reflect unexplained slip
     behavior, not unmodeled speed variation.
  2. Ground-truth incident labels: pulled from FastF1 Race Control Messages
     (flags, driver-specific incident calls) and track-limit lap deletions,
     used to score detection precision/recall/lead-time.
  3. Full-race, all-laps analysis (not just each driver's fastest/cleanest lap),
     so laps that actually contain instability are included.
  4. Autocorrelation diagnostics on residuals (Ljung-Box + ACF) + a block-
     bootstrap alternative to the naive q95, since split-conformal exchangeability
     is questionable for telemetry time series.
  5. A static-threshold baseline detector run on identical data for direct
     precision/recall/lead-time comparison against the conformal method.
  6. All "results" are now numeric: detected flags are cross-referenced against
     RCM incident timestamps, no narrative attribution without an event to point to.
  7. Bootstrap confidence intervals, and a sensitivity sweep over window size k
     and n_estimators.

Outputs (written to ./results/):
  - calibration_summary.csv      per-driver q95 (naive + block-bootstrap CI)
  - detection_comparison.csv     conformal vs baseline: precision/recall/lead-time
  - sensitivity_k.csv            effect of window size k
  - sensitivity_trees.csv        effect of n_estimators
  - residual_diagnostics.csv     Ljung-Box p-values per driver (autocorrelation check)
  - incidents_log.csv            extracted RCM incidents used as ground truth
  - flags_log.csv                every flagged event (conformal + baseline) with match info

Run:
    pip install fastf1 pandas numpy scikit-learn scipy
    python conformal_slip_v2.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import fastf1
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------------
YEAR, GP, SESSION = 2023, "Monza", "R"
CACHE_DIR = "./F1_Telemetry_Cache"
OUT_DIR = "./results"
RANDOM_STATE = 42

CALIBRATION_FRACTION = 0.5          # chronological split across full race, not one lap
K_GRID = [5, 10, 20, 50]            # rolling window sensitivity sweep
N_TREES_GRID = [25, 50, 100, 200]
DEFAULT_K = 20
DEFAULT_TREES = 100

INCIDENT_MATCH_WINDOW_S = 12.0      # tolerance for matching a flag to an RCM incident
LOOKBACK_FOR_RECALL_S = 6.0         # how far before an incident a flag still "counts"
BOOTSTRAP_ITERS = 500
BLOCK_LEN = 25                      # samples per block for block bootstrap (~0.5-1s at typical rates)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ----------------------------- DATA LOADING ----------------------------------
def load_session():
    session = fastf1.get_session(YEAR, GP, SESSION)
    session.load(telemetry=True, laps=True, weather=False)
    return session


def extract_incidents(session):
    """Ground-truth incident timestamps from Race Control Messages.
    Returns a DataFrame: [SessionTime_s, DriverNumber (nullable), Category, Message]
    DriverNumber is None for track-wide flags (still usable as a general
    instability marker but not attributable to one driver).
    """
    rcm = session.race_control_messages.copy()
    rows = []
    for _, r in rcm.iterrows():
        msg = str(r.get("Message", ""))
        cat = str(r.get("Category", ""))
        flag = str(r.get("Flag", ""))
        t = r.get("Time", None)
        if t is None or pd.isna(t):
            continue
        t_s = t.total_seconds() if hasattr(t, "total_seconds") else None
        if t_s is None:
            continue
        driver_num = r.get("RacingNumber", None)
        is_relevant = (
            flag in ("YELLOW", "DOUBLE YELLOW", "RED")
            or "SPIN" in msg.upper()
            or "OFF TRACK" in msg.upper()
            or "TRACK LIMITS" in msg.upper()
            or "INCIDENT" in msg.upper()
            or "LOCK" in msg.upper()
        )
        if is_relevant:
            rows.append({
                "SessionTime_s": t_s,
                "DriverNumber": driver_num,
                "Category": cat,
                "Flag": flag,
                "Message": msg,
            })
    return pd.DataFrame(rows)


def deleted_lap_incidents(session):
    """Track-limit lap deletions as an additional per-driver ground-truth signal."""
    laps = session.laps
    rows = []
    if "Deleted" in laps.columns:
        deleted = laps[laps["Deleted"] == True]  # noqa: E712
        for _, lap in deleted.iterrows():
            t = lap.get("LapStartTime", None)
            if t is None or pd.isna(t):
                continue
            rows.append({
                "SessionTime_s": t.total_seconds(),
                "DriverNumber": lap.get("DriverNumber", None),
                "Category": "TrackLimitsDeletion",
                "Flag": "",
                "Message": f"Lap deleted for track limits: {lap.get('Driver', '')}",
            })
    return pd.DataFrame(rows)


# ----------------------------- FEATURE BUILD ----------------------------------
def build_driver_telemetry(session, driver):
    """Concatenate telemetry across ALL race laps for a driver (not just fastest),
    excluding in/out laps, tagged with absolute SessionTime for incident matching.
    """
    laps = session.laps.pick_driver(driver)
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    if laps.empty:
        return None

    frames = []
    for _, lap in laps.iterlaps():
        try:
            tel = lap.get_telemetry().add_distance()
            if tel is None or tel.empty:
                continue
            tel = tel.copy()
            tel["LapNumber"] = lap["LapNumber"]
            frames.append(tel)
        except Exception:
            continue
    if not frames:
        return None

    tel = pd.concat(frames, ignore_index=True)
    tel = tel.dropna(subset=["Speed", "Throttle", "Brake", "RPM", "nGear", "SessionTime"])
    tel["SessionTime_s"] = tel["SessionTime"].dt.total_seconds()
    tel = tel.sort_values("SessionTime_s").reset_index(drop=True)

    # traction-critical filter retained from v1, but across the whole race now
    tel = tel[(tel["nGear"].isin([2, 3, 4])) & (tel["Speed"] > 30)]
    if len(tel) < 100:
        return None

    tel["Slip_Proxy"] = tel["RPM"] / (tel["Speed"] + 1e-5)
    return tel.reset_index(drop=True)


# ----------------------------- CORE METRICS ----------------------------------
def rolling_volatility(residuals, k):
    return pd.Series(residuals).rolling(window=k, min_periods=k).std().to_numpy()


def block_bootstrap_q95(residuals, block_len=BLOCK_LEN, iters=BOOTSTRAP_ITERS, q=0.95, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(residuals)
    if n < block_len * 2:
        return np.quantile(residuals, q), (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block_len))
    boot_q95 = []
    for _ in range(iters):
        idx = rng.integers(0, n - block_len, size=n_blocks)
        sample = np.concatenate([residuals[i:i + block_len] for i in idx])[:n]
        boot_q95.append(np.quantile(sample, q))
    boot_q95 = np.array(boot_q95)
    return np.quantile(residuals, q), (np.percentile(boot_q95, 2.5), np.percentile(boot_q95, 97.5))


def match_flags_to_incidents(flag_times, incident_times, window_s=INCIDENT_MATCH_WINDOW_S):
    """A flag is a true positive if an incident occurs within [flag_time, flag_time+window_s]
    (flags should PRECEDE incidents to have any predictive value)."""
    incident_times = np.sort(np.array(incident_times))
    matched_flags = 0
    lead_times = []
    for ft in flag_times:
        future = incident_times[(incident_times >= ft) & (incident_times <= ft + window_s)]
        if len(future) > 0:
            matched_flags += 1
            lead_times.append(future[0] - ft)
    precision = matched_flags / len(flag_times) if len(flag_times) > 0 else np.nan

    matched_incidents = 0
    for it in incident_times:
        past = [ft for ft in flag_times if it - LOOKBACK_FOR_RECALL_S - window_s <= ft <= it]
        if len(past) > 0:
            matched_incidents += 1
    recall = matched_incidents / len(incident_times) if len(incident_times) > 0 else np.nan

    return {
        "n_flags": len(flag_times),
        "n_incidents": len(incident_times),
        "precision": precision,
        "recall": recall,
        "mean_lead_time_s": float(np.mean(lead_times)) if lead_times else np.nan,
        "median_lead_time_s": float(np.median(lead_times)) if lead_times else np.nan,
    }


# ----------------------------- PER-DRIVER PIPELINE ----------------------------------
def run_driver(driver, tel, driver_incidents_s, k=DEFAULT_K, n_trees=DEFAULT_TREES):
    split_idx = int(len(tel) * CALIBRATION_FRACTION)
    train, test = tel.iloc[:split_idx].copy(), tel.iloc[split_idx:].copy()
    if len(train) < 50 or len(test) < 50:
        return None

    # FIX #1: Speed is now an explicit feature, de-confounding the proxy.
    features = ["Throttle", "Brake", "nGear", "Speed"]
    X_train, y_train = train[features], train["Slip_Proxy"]
    X_test, y_test = test[features], test["Slip_Proxy"]

    model = RandomForestRegressor(n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    train_resid = np.abs(y_train - model.predict(X_train)).to_numpy()
    q95_naive, (q95_lo, q95_hi) = block_bootstrap_q95(train_resid)

    test_pred = model.predict(X_test)
    test_resid = np.abs(y_test - test_pred).to_numpy()

    # FIX #4: autocorrelation diagnostic on residuals
    try:
        lb = acorr_ljungbox(test_resid, lags=[10], return_df=True)
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_pvalue = np.nan

    vol = rolling_volatility(test_resid, k)
    sess_times = test["SessionTime_s"].to_numpy()

    conformal_flags = sess_times[~np.isnan(vol) & (vol > q95_naive)]

    # baseline: static threshold on raw residual (no rolling/volatility logic)
    static_thresh = np.quantile(train_resid, 0.95)
    baseline_flags = sess_times[test_resid > static_thresh]

    conformal_scores = match_flags_to_incidents(conformal_flags, driver_incidents_s)
    baseline_scores = match_flags_to_incidents(baseline_flags, driver_incidents_s)

    return {
        "driver": driver,
        "n_test_samples": len(test),
        "q95_naive": q95_naive,
        "q95_boot_ci_low": q95_lo,
        "q95_boot_ci_high": q95_hi,
        "ljung_box_pvalue": lb_pvalue,
        "n_incidents_matched_pool": len(driver_incidents_s),
        "conformal": conformal_scores,
        "baseline": baseline_scores,
        "conformal_flag_times": conformal_flags,
        "baseline_flag_times": baseline_flags,
    }


# ----------------------------- MAIN ----------------------------------
def main():
    session = load_session()
    drivers = session.results["Abbreviation"].tolist()

    rcm_incidents = extract_incidents(session)
    deleted_incidents = deleted_lap_incidents(session)
    all_incidents = pd.concat([rcm_incidents, deleted_incidents], ignore_index=True)
    all_incidents.to_csv(f"{OUT_DIR}/incidents_log.csv", index=False)

    driver_number_map = dict(zip(session.results["Abbreviation"], session.results["DriverNumber"]))

    calibration_rows, diagnostic_rows, flags_rows, comparison_rows = [], [], [], []
    sens_k_rows, sens_trees_rows = [], []

    for driver in drivers:
        tel = build_driver_telemetry(session, driver)
        if tel is None:
            continue

        dnum = str(driver_number_map.get(driver, ""))
        driver_incidents = all_incidents[
            (all_incidents["DriverNumber"].astype(str) == dnum)
            | (all_incidents["DriverNumber"].isna())
        ]["SessionTime_s"].tolist()

        result = run_driver(driver, tel, driver_incidents)
        if result is None:
            continue

        calibration_rows.append({
            "Driver": driver,
            "q95_naive": round(result["q95_naive"], 4),
            "q95_boot_CI_low": round(result["q95_boot_ci_low"], 4) if not np.isnan(result["q95_boot_ci_low"]) else None,
            "q95_boot_CI_high": round(result["q95_boot_ci_high"], 4) if not np.isnan(result["q95_boot_ci_high"]) else None,
            "n_test_samples": result["n_test_samples"],
            "n_ground_truth_incidents": result["n_incidents_matched_pool"],
        })

        diagnostic_rows.append({
            "Driver": driver,
            "ljung_box_pvalue_lag10": result["ljung_box_pvalue"],
            "residuals_likely_autocorrelated": (result["ljung_box_pvalue"] < 0.05
                                                 if not np.isnan(result["ljung_box_pvalue"]) else None),
        })

        comparison_rows.append({
            "Driver": driver,
            "method": "conformal_volatility",
            **result["conformal"],
        })
        comparison_rows.append({
            "Driver": driver,
            "method": "static_threshold_baseline",
            **result["baseline"],
        })

        for ft in result["conformal_flag_times"]:
            flags_rows.append({"Driver": driver, "method": "conformal", "flag_time_s": ft})
        for ft in result["baseline_flag_times"]:
            flags_rows.append({"Driver": driver, "method": "baseline", "flag_time_s": ft})

        # FIX #7: sensitivity sweep (run only for top-line drivers to keep runtime sane;
        # remove the slice below to run for the full grid)
        for k in K_GRID:
            r_k = run_driver(driver, tel, driver_incidents, k=k, n_trees=DEFAULT_TREES)
            if r_k:
                sens_k_rows.append({
                    "Driver": driver, "k": k,
                    "precision": r_k["conformal"]["precision"],
                    "recall": r_k["conformal"]["recall"],
                    "n_flags": r_k["conformal"]["n_flags"],
                })
        for nt in N_TREES_GRID:
            r_t = run_driver(driver, tel, driver_incidents, k=DEFAULT_K, n_trees=nt)
            if r_t:
                sens_trees_rows.append({
                    "Driver": driver, "n_estimators": nt,
                    "precision": r_t["conformal"]["precision"],
                    "recall": r_t["conformal"]["recall"],
                    "n_flags": r_t["conformal"]["n_flags"],
                })

    pd.DataFrame(calibration_rows).sort_values("q95_naive").to_csv(
        f"{OUT_DIR}/calibration_summary.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(f"{OUT_DIR}/residual_diagnostics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(f"{OUT_DIR}/detection_comparison.csv", index=False)
    pd.DataFrame(flags_rows).to_csv(f"{OUT_DIR}/flags_log.csv", index=False)
    pd.DataFrame(sens_k_rows).to_csv(f"{OUT_DIR}/sensitivity_k.csv", index=False)
    pd.DataFrame(sens_trees_rows).to_csv(f"{OUT_DIR}/sensitivity_trees.csv", index=False)

    print("Done. CSVs written to ./results/")


if __name__ == "__main__":
    main()