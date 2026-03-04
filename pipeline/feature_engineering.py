"""
feature_engineering.py — IDENTIK dengan training notebook
Semua konstanta, fungsi build_snapshot, engineer_features, dan
build_features_for_inference harus sama persis dengan cell [1] notebook.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ── Konstanta (dari notebook cell [2]) ────────────────────────────────────
COUNTER_COLS = ["171_0", "666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]
HIST_COLS = {
    "167": [f"167_{i}" for i in range(10)],
    "272": [f"272_{i}" for i in range(10)],
    "291": [f"291_{i}" for i in range(11)],
    "158": [f"158_{i}" for i in range(10)],
    "459": [f"459_{i}" for i in range(20)],
    "397": [f"397_{i}" for i in range(36)],
}
ZERO_INFLATED_COLS = ["309_0", "370_0"]
DROP_MULTICOLINEAR = ["835_0"]


def build_snapshot(ops_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build last-readout snapshot + aggregate statistics per vehicle.
    IDENTIK dengan build_snapshot() di notebook cell [1].
    """
    ops_sorted = ops_df.sort_values(["vehicle_id", "time_step"])

    # Last readout per vehicle
    last = ops_sorted.groupby("vehicle_id").last().reset_index()
    last.columns = [
        "vehicle_id" if c == "vehicle_id" else f"last_{c}" for c in last.columns
    ]

    # Aggregate statistics (mean, std, max) untuk COUNTER_COLS
    agg_dict = {
        col: ["mean", "std", "max"] for col in COUNTER_COLS if col in ops_sorted.columns
    }
    stats_df = ops_sorted.groupby("vehicle_id").agg(agg_dict).reset_index()
    stats_df.columns = [
        "vehicle_id" if lv0 == "vehicle_id" else f"{lv1}_{lv0}"
        for lv0, lv1 in stats_df.columns.to_flat_index()
    ]

    # Readout count per vehicle
    readout_cnt = (
        ops_sorted.groupby("vehicle_id").size().reset_index(name="readout_count")
    )

    snap = last.merge(stats_df, on="vehicle_id", how="left").merge(
        readout_cnt, on="vehicle_id", how="left"
    )
    return snap


def engineer_features(snap: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer rate, log-transform, and activity features.
    IDENTIK dengan engineer_features() di notebook cell [1].
    """
    df = snap.copy()
    last_ts = df["last_time_step"].clip(lower=0) + 1

    # Rate features: counter / (time_step + 1)
    for col in COUNTER_COLS:
        lc, mc = f"last_{col}", f"mean_{col}"
        if lc in df.columns:
            df[f"rate_{col}"] = df[lc] / last_ts
        if mc in df.columns:
            df[f"mean_rate_{col}"] = df[mc] / last_ts

    # Zero-inflated binary indicators
    for col in ZERO_INFLATED_COLS:
        lc = f"last_{col}"
        if lc in df.columns:
            df[f"{col}_active"] = (df[lc] > 0).astype(np.int8)

    # Log1p transform pada fitur numerik
    log_targets = [
        f"{pfx}{col}"
        for col in COUNTER_COLS
        for pfx in ("last_", "mean_", "max_", "rate_", "mean_rate_", "std_")
        if f"{pfx}{col}" in df.columns
    ]
    for c in log_targets:
        df[c] = np.log1p(df[c].clip(lower=0).fillna(0))

    # Fill NaN di histogram last-readout features
    hist_last = [f"last_{c}" for c in HIST_COLS if f"last_{c}" in df.columns]
    df[hist_last] = df[hist_last].fillna(0)

    return df


def build_features_for_inference(
    ops_df: pd.DataFrame,
    specs_df: pd.DataFrame,
    encoder: OrdinalEncoder,
    feature_cols: list,
) -> tuple:
    """
    Pipeline feature engineering lengkap untuk inference.
    Mereplikasi urutan PERSIS sama dengan notebook:
      build_snapshot → engineer_features → merge specs → encode → drop multicolinear → align

    Returns:
        snap (DataFrame, berisi vehicle_id & last_time_step)
        X    (DataFrame, kolom = feature_cols, sudah NaN-filled, belum discale)
    """
    # Step 1: Snapshot
    snap = build_snapshot(ops_df)

    # Step 2: Engineer features
    snap = engineer_features(snap)

    # Step 3: Merge specifications
    snap = snap.merge(specs_df, on="vehicle_id", how="left")

    # Step 4: Encode categorical spec columns
    spec_cols = [c for c in specs_df.columns if c != "vehicle_id"]
    avail_spec = [c for c in spec_cols if c in snap.columns]
    for col in avail_spec:
        snap[col] = snap[col].astype(str)
    snap[avail_spec] = encoder.transform(snap[avail_spec])

    # Step 5: Drop multicolinear
    drop_cols = [
        f"{pfx}{col}"
        for col in DROP_MULTICOLINEAR
        for pfx in ("last_", "mean_", "std_", "max_", "rate_", "mean_rate_")
        if f"{pfx}{col}" in snap.columns
    ]
    snap = snap.drop(columns=drop_cols, errors="ignore")

    # Step 6: Align ke feature_cols (isi kolom yang hilang dengan 0)
    for c in feature_cols:
        if c not in snap.columns:
            snap[c] = 0.0

    X = snap[feature_cols].fillna(0)
    return snap, X
