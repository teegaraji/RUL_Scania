import os
import sys

# Must be set before any pycox import — Streamlit Cloud venv is read-only
os.environ.setdefault("PYCOX_DATA_DIR", "/tmp/pycox_data")

import numpy as np
import pandas as pd
import streamlit as st

# Pastikan folder deployment/ ada di sys.path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.feature_engineering import build_features_for_inference
from pipeline.predict import load_models, predict

# ── Cost matrix (IDENTIK dengan training notebook) ──────────────────────────
COST_CORRECT = np.array(
    [
        [0, 0, 0, 7, 500],  # pred=0
        [7, 0, 2, 2, 200],  # pred=1
        [8, 7, 0, 2, 200],  # pred=2
        [9, 8, 7, 0, 200],  # pred=3
        [10, 9, 8, 7, 0],  # pred=4
    ],
    dtype=np.float32,
)


def compute_cost(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    return int(COST_CORRECT[y_pred.astype(int), y_true.astype(int)].sum())


def per_vehicle_actual_cost(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Cost per kendaraan berdasarkan label aktual."""
    return COST_CORRECT[y_pred.astype(int), y_true.astype(int)]


def per_vehicle_risk_cost(y_pred: np.ndarray) -> np.ndarray:
    """
    Risk cost = worst-case cost jika prediksi ternyata salah.
    Dipakai ketika labels tidak tersedia (deployment nyata).
    Rumus: max cost dari semua kemungkinan true label untuk prediksi ini.
    """
    return np.array([COST_CORRECT[p, :].max() for p in y_pred.astype(int)])


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SCANIA Fleet Monitor", page_icon="🚛", layout="wide")
st.title("🚛 SCANIA Fleet — Prediksi Kegagalan Komponen X")

st.markdown(
    """
    Upload file CSV dari sistem telematika kendaraan. File **labels** bersifat opsional —
    jika tersedia, aplikasi akan menghitung **cost aktual**. Jika tidak, akan ditampilkan
    **risk cost estimasi** (worst-case).
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    ops_file = st.file_uploader("📂 Operational Readouts (.csv)", type="csv")
with col2:
    spec_file = st.file_uploader("📂 Specifications (.csv)", type="csv")
with col3:
    labels_file = st.file_uploader(
        "📋 Labels (.csv) — opsional, untuk hitung cost aktual",
        type="csv",
        help="Kolom yang dibutuhkan: vehicle_id, class_label",
    )

if ops_file and spec_file:
    ops_df = pd.read_csv(ops_file)
    spec_df = pd.read_csv(spec_file)
    labels_df = pd.read_csv(labels_file) if labels_file else None
    eval_mode = labels_df is not None

    st.info(
        f"{'🔬 Mode Evaluasi' if eval_mode else '🚀 Mode Deployment'} — "
        f"Operational: **{len(ops_df):,} baris**, "
        f"Specs: **{len(spec_df):,} kendaraan**"
        + (f", Labels: **{len(labels_df):,} kendaraan**" if eval_mode else "")
    )

    # 1. Load model
    @st.cache_resource
    def get_models():
        return load_models()

    with st.spinner("Memuat model..."):
        cox, dh, scaler, encoder, feature_cols = get_models()

    # 2. Feature engineering
    with st.spinner("Memproses fitur..."):
        snap, X = build_features_for_inference(ops_df, spec_df, encoder, feature_cols)
        last_ts = snap["last_time_step"].values

    # 3. Prediksi
    with st.spinner("Menghitung prediksi ensemble (CoxPH + DeepHit)..."):
        rul, classes = predict(X, last_ts, cox, dh, scaler, feature_cols)

    # 4. Susun hasil
    result = snap[["vehicle_id"]].copy()
    result["Prediksi RUL"] = rul.round(1)
    result["Kelas Prediksi"] = classes
    result["Keterangan"] = result["Kelas Prediksi"].map(
        {
            0: "✅ Normal",
            1: "🟡 Perlu Perhatian",
            2: "🟠 Segera Ditangani",
            3: "🔴 Kritis",
            4: "⛔ Sangat Kritis",
        }
    )

    # 5. Hitung cost
    y_pred_arr = classes.astype(int)

    if eval_mode:
        # Merge label aktual
        merged = result.merge(
            labels_df[["vehicle_id", "class_label"]].rename(
                columns={"class_label": "Kelas Aktual"}
            ),
            on="vehicle_id",
            how="left",
        )
        y_true_arr = merged["Kelas Aktual"].fillna(0).astype(int).values
        merged["Cost per Kendaraan"] = per_vehicle_actual_cost(
            y_pred_arr, y_true_arr
        ).astype(int)
        total_cost = compute_cost(y_pred_arr, y_true_arr)
        baseline_cost = compute_cost(np.zeros(len(y_true_arr), dtype=int), y_true_arr)
        saving_pct = (baseline_cost - total_cost) / baseline_cost * 100
        result = merged
    else:
        result["Risk Cost (worst-case)"] = per_vehicle_risk_cost(y_pred_arr).astype(int)
        total_cost = None
        baseline_cost = None
        saving_pct = None

    # 6. Metrik ringkasan
    n_total = len(result)
    n_critical = int((result["Kelas Prediksi"] >= 3).sum())
    n_warning = int((result["Kelas Prediksi"] == 2).sum())
    n_ok = int((result["Kelas Prediksi"] <= 1).sum())

    st.divider()
    st.subheader("Ringkasan")

    if eval_mode:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🚛 Total Kendaraan", f"{n_total:,}")
        m2.metric("⛔ Kritis (Kelas 3–4)", n_critical)
        m3.metric("🟠 Perlu Tindakan (Kelas 2)", n_warning)
        m4.metric("✅ Normal / Aman", n_ok)
        m5.metric("💰 Total Cost Model", f"{total_cost:,}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🚛 Total Kendaraan", f"{n_total:,}")
        m2.metric("⛔ Kritis (Kelas 3–4)", n_critical)
        m3.metric("🟠 Perlu Tindakan (Kelas 2)", n_warning)
        m4.metric("✅ Normal / Aman", n_ok)
        st.caption(
            "ℹ️ Upload file labels untuk menghitung cost aktual. "
            "Kolom **Risk Cost (worst-case)** di tabel menunjukkan biaya maksimal "
            "yang terjadi jika prediksi salah untuk setiap kendaraan."
        )

    # 7. Penjelasan cost matrix (expandable)
    with st.expander("ℹ️ Cara membaca Cost Matrix"):
        st.markdown(
            """
            Cost dihitung berdasarkan **COST_CORRECT[prediksi][aktual]**:

            | Pred \\ Aktual | Kelas 0 | Kelas 1 | Kelas 2 | Kelas 3 | Kelas 4 |
            |--------------|---------|---------|---------|---------|---------|
            | **Pred 0** | 0 | 0 | 0 | 7 | **500** |
            | **Pred 1** | 7 | 0 | 2 | 2 | 200 |
            | **Pred 2** | 8 | 7 | 0 | 2 | 200 |
            | **Pred 3** | 9 | 8 | 7 | 0 | 200 |
            | **Pred 4** | 10 | 9 | 8 | 7 | 0 |

            - **Cost 500**: kendaraan Kelas 4 (sangat kritis) diprediksi **Kelas 0** → *miss* paling berbahaya
            - **Cost 0**: prediksi tepat
            - **Over-predict** (prediksi terlalu tinggi) → cost kecil tapi ada biaya servis tidak perlu
            - **Under-predict** (prediksi terlalu rendah) → cost besar, kendaraan kritis tidak tertangani
            """
        )

    # 8. Tabel detail
    st.divider()
    st.subheader(f"Detail {n_total:,} Kendaraan")
    sort_col = "Cost per Kendaraan" if eval_mode else "Risk Cost (worst-case)"
    st.dataframe(
        result.sort_values(sort_col, ascending=False),
        use_container_width=True,
    )

    # 9. Distribusi kelas (bar chart)
    with st.expander("📊 Lihat Distribusi Kelas Prediksi"):
        dist = result["Kelas Prediksi"].value_counts().sort_index().reset_index()
        dist.columns = ["Kelas", "Jumlah"]
        dist["Label"] = dist["Kelas"].map(
            {
                0: "0-Normal",
                1: "1-Perhatian",
                2: "2-Segera",
                3: "3-Kritis",
                4: "4-SangatKritis",
            }
        )
        st.bar_chart(dist.set_index("Label")["Jumlah"])

    # 10. Peringatan / konfirmasi
    if n_critical > 0:
        st.error(f"⚠️ {n_critical} kendaraan memerlukan perhatian SEGERA (Kelas ≥ 3)")
    elif n_warning > 0:
        st.warning(f"🔔 {n_warning} kendaraan perlu dijadwalkan servis (Kelas 2)")
    else:
        st.success(
            "Semua kendaraan dalam kondisi normal atau dapat ditangani terjadwal."
        )

    # 11. Download hasil
    csv_out = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Hasil Prediksi (.csv)",
        data=csv_out,
        file_name="prediksi_rul.csv",
        mime="text/csv",
    )

elif ops_file and not spec_file:
    st.warning("⚠️ File Specifications belum diupload.")
elif spec_file and not ops_file:
    st.warning("⚠️ File Operational Readouts belum diupload.")
