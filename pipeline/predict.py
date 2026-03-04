"""
predict.py — Inference pipeline untuk Ensemble (CoxPH + DeepHit)
Semua parameter HARUS identik dengan training notebook.
"""

import os

# Must be set BEFORE importing pycox — Streamlit Cloud mounts the venv
# read-only, so pycox cannot create its data dir inside site-packages.
os.environ.setdefault("PYCOX_DATA_DIR", "/tmp/pycox_data")

import joblib
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle  # ← DeepHitSingle, bukan DeepHit

# ── Konstanta ensemble (dari notebook cell [8]) ────────────────────────────
W_COX = 0.6
SHIFT = 16
MAX_RUL = 600.0

# Path artifacts relatif terhadap file ini (deployment/pipeline/predict.py)
#   → deployment/artifacts/
_ARTIFACTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts")
)


def load_models(artifacts_dir: str = _ARTIFACTS_DIR):
    """
    Load semua artefak yang dibutuhkan:
      coxph_model.pkl  → CoxPHFitter (lifelines)
      scaler.pkl       → StandardScaler (fit pada X_surv_train)
      encoder.pkl      → OrdinalEncoder (fit pada spec columns)
      feature_cols.pkl → list FEATURE_COLS (urutan kolom saat training)
      labtrans.pkl     → DeepHitSingle label transform
      deephit_net.pt   → state_dict MLP DeepHit
    """
    cox = joblib.load(os.path.join(artifacts_dir, "coxph_model.pkl"))
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
    encoder = joblib.load(os.path.join(artifacts_dir, "encoder.pkl"))
    feature_cols = joblib.load(os.path.join(artifacts_dir, "feature_cols.pkl"))
    labtrans = joblib.load(os.path.join(artifacts_dir, "labtrans.pkl"))

    # Rebuild MLP — arsitektur HARUS identik dengan training:
    #   in_features = len(FEATURE_COLS), nodes=[256,128,64], dropout=0.2
    net = tt.practical.MLPVanilla(
        len(feature_cols),  # ← dari feature_cols.pkl, bukan hardcode 150
        [256, 128, 64],
        labtrans.out_features,
        batch_norm=True,
        dropout=0.2,  # ← 0.2, BUKAN 0.1
    )
    net.load_state_dict(
        torch.load(
            os.path.join(artifacts_dir, "deephit_net.pt"),
            map_location="cpu",
        )
    )
    dh = DeepHitSingle(  # ← DeepHitSingle, bukan DeepHit
        net,
        tt.optim.Adam,
        alpha=0.2,
        sigma=0.1,
        duration_index=labtrans.cuts,
    )
    return cox, dh, scaler, encoder, feature_cols


def extract_median_survival(surv_df: pd.DataFrame) -> np.ndarray:
    """
    Ekstrak median survival time dari survival function DataFrame.
    IDENTIK dengan extract_median_survival() di notebook cell [4].
    """
    medians = []
    for col in surv_df.columns:
        s = surv_df[col]
        below_50 = s[s <= 0.5]
        if len(below_50) > 0:
            medians.append(below_50.index[0])
        else:
            medians.append(s.index[-1])  # tidak pernah turun ke 0.5 → waktu max
    return np.array(medians)


def rul_to_class(rul: float) -> int:
    """
    Konversi RUL numerik → kelas 0-4.
    IDENTIK dengan rul_to_class() di notebook cell [2].
    """
    if rul > 48:
        return 0
    elif rul > 24:
        return 1
    elif rul > 12:
        return 2
    elif rul > 6:
        return 3
    else:
        return 4


def predict(
    X_raw: pd.DataFrame,
    last_ts: np.ndarray,
    cox,
    dh,
    scaler,
    feature_cols: list,
) -> tuple:
    """
    Jalankan ensemble CoxPH + DeepHit.

    Args:
        X_raw       : DataFrame kolom = feature_cols (belum discale)
        last_ts     : array last_time_step per kendaraan, shape (N,)
        cox         : CoxPHFitter dari lifelines
        dh          : DeepHitSingle dari pycox
        scaler      : StandardScaler yang di-fit saat training
        feature_cols: list nama kolom sesuai urutan training

    Returns:
        rul_ens : np.ndarray RUL prediksi ensemble
        classes : np.ndarray kelas 0-4 per kendaraan
    """
    # Scale fitur — WAJIB sama persis seperti saat training
    X_scaled = pd.DataFrame(
        scaler.transform(X_raw),
        columns=feature_cols,
        index=X_raw.index,
    )

    # ── CoxPH ──────────────────────────────────────────────────────────────
    median_cox = cox.predict_median(X_scaled)
    median_cox = median_cox.replace([np.inf, -np.inf], MAX_RUL).fillna(MAX_RUL).values
    # RUL = predicted survival time − last observed time_step
    rul_cox = np.clip(median_cox - last_ts, 0, None)

    # ── DeepHit ────────────────────────────────────────────────────────────
    X_dh = X_scaled.values.astype("float32")
    surv_df = dh.predict_surv_df(X_dh)  # survival function S(t)
    median_dh_raw = extract_median_survival(surv_df)
    rul_dh = np.clip(median_dh_raw - last_ts, 0, None)

    # ── Ensemble ───────────────────────────────────────────────────────────
    rul_ens = W_COX * rul_cox + (1.0 - W_COX) * rul_dh + SHIFT
    classes = np.array([rul_to_class(r) for r in rul_ens])

    return rul_ens, classes
