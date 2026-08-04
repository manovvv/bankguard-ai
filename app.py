import io
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# =============================================================================
# 1. KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="BankGuard AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# 2. SESSION STATE (theme, log simulasi, threshold global)
# =============================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "simulation_log" not in st.session_state:
    st.session_state.simulation_log = []  # list of dict
if "risk_threshold" not in st.session_state:
    st.session_state.risk_threshold = (30, 70)  # (low, high) dalam persen

# =============================================================================
# 3. CUSTOM CSS ADAPTIF (Dark/Light + Responsive Mobile) + Micro-interactions
# =============================================================================
def inject_css(theme: str):
    if theme == "dark":
        bg_grad = "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)"
        card_bg = "rgba(255,255,255,0.04)"
        card_border = "rgba(255,255,255,0.10)"
        text_primary = "#ffffff"
        text_muted = "#94a3b8"
        app_bg = "#0b1020"
    else:
        bg_grad = "linear-gradient(135deg, #eef2ff 0%, #f8fafc 50%, #eef2ff 100%)"
        card_bg = "rgba(15,23,42,0.03)"
        card_border = "rgba(15,23,42,0.10)"
        text_primary = "#0f172a"
        text_muted = "#475569"
        app_bg = "#f8fafc"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {app_bg}; }}

        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .hero-header {{
            background: {bg_grad};
            border: 1px solid {card_border};
            padding: 16px 20px;
            border-radius: 14px;
            color: #ffffff;
            margin-bottom: 15px;
            box-shadow: 0 8px 20px -5px rgba(0, 0, 0, 0.3);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .hero-title-group {{ display: flex; align-items: center; gap: 12px; }}
        .hero-icon {{
            font-size: 28px; background: rgba(255,255,255,0.1);
            padding: 8px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.15);
        }}
        .hero-title {{
            font-size: 22px; font-weight: 800; margin: 0; letter-spacing: -0.5px;
            background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .hero-subtitle {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}
        .status-badge {{
            background: rgba(16,185,129,0.15); color: #34d399;
            border: 1px solid rgba(16,185,129,0.3); padding: 4px 10px;
            border-radius: 20px; font-size: 11px; font-weight: 600;
            display: flex; align-items: center; gap: 6px;
        }}
        .status-dot {{
            width: 6px; height: 6px; background-color: #34d399;
            border-radius: 50%; box-shadow: 0 0 6px #34d399;
            animation: pulse-dot 1.6s infinite;
        }}
        @keyframes pulse-dot {{
            0% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} 100% {{ opacity: 1; }}
        }}

        /* Kartu generik (KPI, panel What-If, dsb) */
        .ui-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 14px;
            padding: 16px 18px;
            color: {text_primary};
        }}
        .ui-card .label {{
            font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
            color: {text_muted}; margin-bottom: 4px;
        }}
        .ui-card .value {{ font-size: 24px; font-weight: 700; color: {text_primary}; }}

        /* Alert berkedip untuk risiko ekstrem */
        .pulse-alert {{
            border: 1px solid #ef4444;
            background: rgba(239,68,68,0.12);
            border-radius: 12px;
            padding: 14px 18px;
            color: #fca5a5;
            font-weight: 600;
            animation: pulse-alert 1.2s infinite;
        }}
        @keyframes pulse-alert {{
            0% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.45); }}
            70% {{ box-shadow: 0 0 0 12px rgba(239,68,68,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0); }}
        }}

        /* Navigasi radio button responsive (swipeable di HP) */
        div[role="radiogroup"] {{
            display: flex !important; flex-direction: row !important;
            flex-wrap: nowrap !important; overflow-x: auto !important;
            -webkit-overflow-scrolling: touch; gap: 8px !important;
            background-color: rgba(125,125,125,0.1) !important;
            padding: 6px !important; border-radius: 12px !important;
            border: 1px solid rgba(125,125,125,0.2) !important;
            scrollbar-width: none;
        }}
        div[role="radiogroup"]::-webkit-scrollbar {{ display: none; }}
        div[role="radiogroup"] label {{
            white-space: nowrap !important; flex: 0 0 auto !important;
            background-color: transparent !important; padding: 6px 14px !important;
            border-radius: 8px !important; font-size: 13px !important; font-weight: 600 !important;
        }}

        /* Tabs styling */
        button[data-baseweb="tab"] {{ font-weight: 600 !important; }}

        @media (max-width: 768px) {{
            .hero-header {{ padding: 12px 15px; border-radius: 10px; }}
            .hero-title {{ font-size: 18px; }}
            .hero-subtitle {{ font-size: 11px; }}
            .hero-icon {{ font-size: 22px; padding: 6px; }}
            .status-badge {{ font-size: 10px; padding: 3px 8px; }}
            .ui-card .value {{ font-size: 20px; }}
        }}
    </style>
    """, unsafe_allow_html=True)


inject_css(st.session_state.theme)

# =============================================================================
# 4. LOAD ARTIFACTS (tidak diubah dari kode asli)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path(".")


def find_file(filename):
    root_path = BASE_DIR / filename
    models_path = BASE_DIR / "models" / filename
    if root_path.exists():
        return root_path
    elif models_path.exists():
        return models_path
    return None


@st.cache_resource
def load_artifacts():
    artifacts = {}
    fraud_model_path = find_file("fraud_model.pkl")
    fraud_feat_path = find_file("fraud_features.pkl")
    loan_model_path = find_file("loan_model.pkl")
    loan_feat_path = find_file("loan_features.pkl")

    if fraud_model_path and fraud_feat_path:
        with open(fraud_model_path, "rb") as f:
            artifacts["fraud_model"] = pickle.load(f)
        with open(fraud_feat_path, "rb") as f:
            artifacts["fraud_features"] = pickle.load(f)

    if loan_model_path and loan_feat_path:
        with open(loan_model_path, "rb") as f:
            artifacts["loan_model"] = pickle.load(f)
        with open(loan_feat_path, "rb") as f:
            artifacts["loan_features"] = pickle.load(f)

    return artifacts


artifacts = load_artifacts()


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================
def format_rp(val):
    return f"Rp {val:,.0f}".replace(",", ".")


def get_thresholds():
    low, high = st.session_state.risk_threshold
    return low / 100.0, high / 100.0


def decision_from_prob(prob: float):
    low, high = get_thresholds()
    if prob >= high:
        return "BLOCKED", "🔴"
    elif prob >= low:
        return "REVIEW", "🟡"
    else:
        return "APPROVED", "🟢"


def create_gauge_chart(score, title):
    low, high = get_thresholds()
    color = "#22c55e" if score < low else "#f59e0b" if score < high else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': "%"},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, low * 100], 'color': "rgba(34, 197, 94, 0.2)"},
                {'range': [low * 100, high * 100], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [high * 100, 100], 'color': "rgba(239, 68, 68, 0.2)"},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.85,
                'value': score * 100,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(l=15, r=15, t=35, b=15),
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font_color=st.session_state.theme == "dark" and "#e2e8f0" or "#0f172a")
    return fig


def log_event(module: str, prob: float, decision: str, **inputs):
    """Simpan satu baris hasil simulasi ke log session (bukan setiap slider digeser,
    hanya saat pengguna eksplisit submit / menekan tombol 'Simpan ke Log')."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": module,
        "probability": round(prob, 4),
        "decision": decision,
    }
    entry.update(inputs)
    st.session_state.simulation_log.append(entry)
    st.toast(f"✅ Data tersimpan ke log ({module})", icon="📝")


def extreme_risk_alert(prob: float, label: str = "Transaksi"):
    low, high = get_thresholds()
    if prob >= min(high + 0.15, 0.95):
        st.markdown(
            f'<div class="pulse-alert">🚨 RISIKO EKSTREM TERDETEKSI — {label} ini berada jauh di atas '
            f'ambang batas ({prob:.1%}). Segera lakukan investigasi manual!</div>',
            unsafe_allow_html=True,
        )
        st.snow()


def build_feature_importance(model, features, fallback_weights: dict, label_model: str):
    """Ambil feature_importances_ dari model jika tersedia; jika tidak, gunakan bobot
    rule-based sebagai estimasi pengaruh (diberi label jujur agar tidak menyesatkan)."""
    if model is not None and hasattr(model, "feature_importances_") and features:
        importances = model.feature_importances_
        df_fi = pd.DataFrame({"feature": features, "importance": importances})
        df_fi = df_fi.sort_values("importance", ascending=False).head(15)
        source = f"Model-based ({label_model})"
    else:
        df_fi = pd.DataFrame(
            {"feature": list(fallback_weights.keys()), "importance": list(fallback_weights.values())}
        ).sort_values("importance", ascending=False)
        source = "Rule-based estimate (model tidak menyediakan feature_importances_)"
    return df_fi, source


# =============================================================================
# 6. HERO HEADER + TOGGLE TEMA
# =============================================================================
hc1, hc2 = st.columns([5, 1])
with hc1:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title-group">
            <div class="hero-icon">🛡️</div>
            <div>
                <h1 class="hero-title">BankGuard AI</h1>
                <div class="hero-subtitle">Risk Intelligence Platform — Enterprise SaaS Dashboard</div>
            </div>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            OPERATIONAL
        </div>
    </div>
    """, unsafe_allow_html=True)
with hc2:
    is_dark = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if is_dark else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

# =============================================================================
# 7. PENGATURAN GLOBAL: RISK THRESHOLD (mempengaruhi semua gauge & keputusan)
# =============================================================================
with st.expander("⚙️ Pengaturan Ambang Batas Risiko (berlaku real-time ke semua modul)", expanded=False):
    st.caption(
        "Geser untuk mengatur batas Low/Medium/High Risk. Semua gauge, badge status, dan "
        "keputusan otomatis di bawah akan langsung menyesuaikan tanpa perlu submit ulang."
    )
    st.session_state.risk_threshold = st.slider(
        "Rentang Threshold (%) — [Batas Aman s/d Batas Bahaya]",
        min_value=0, max_value=100, value=st.session_state.risk_threshold, step=1,
    )
    low_pct, high_pct = st.session_state.risk_threshold
    st.caption(f"🟢 Approved: < {low_pct}%  |  🟡 Review: {low_pct}–{high_pct}%  |  🔴 Blocked: > {high_pct}%")

# =============================================================================
# 8. NAVIGASI NAVBAR
# =============================================================================
menu = st.radio(
    label="Pilih Modul Aplikasi:",
    options=["🏠 Dashboard", "🚨 Fraud Simulator", "💳 Credit Risk", "📜 Log & Analytics"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# 9. DASHBOARD UTAMA (KPI dinamis berdasarkan log riil)
# =============================================================================
if menu == "🏠 Dashboard":
    st.subheader("📊 Executive Overview")
    st.caption("Platform Monitoring Portofolio & Sistem Deteksi Anomali AI")

    log_df = pd.DataFrame(st.session_state.simulation_log)
    total_sim = len(log_df)
    blocked_count = int((log_df["decision"] == "BLOCKED").sum()) if total_sim else 0
    approved_count = int((log_df["decision"] == "APPROVED").sum()) if total_sim else 0
    avg_prob = float(log_df["probability"].mean()) if total_sim else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status Sistem", "Aktif 🟢")
    m2.metric("Total Simulasi Tercatat", f"{total_sim}",
              delta=f"+{total_sim}" if total_sim else None)
    m3.metric("Rata-rata Skor Risiko", f"{avg_prob:.1%}" if total_sim else "—",
              delta=None if not total_sim else ("Naik" if avg_prob > 0.5 else "Terkendali"),
              delta_color="inverse" if avg_prob > 0.5 else "normal")
    m4.metric("Transaksi Diblokir", f"{blocked_count}",
              delta=f"{blocked_count}/{total_sim}" if total_sim else None,
              delta_color="inverse" if blocked_count else "normal")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚨 Fraud Risk Engine")
        if "fraud_model" in artifacts:
            st.success("Status: **Ready & Loaded** ✅")
            st.info("Model terkonfigurasi untuk memonitor anomali transaksi real-time.")
        else:
            st.warning("Status: **Model file tidak ditemukan** — mode simulasi tetap berjalan dengan rule-based scoring.")

    with col2:
        st.subheader("💳 Credit Risk Engine")
        if "loan_model" in artifacts:
            st.success("Status: **Ready & Loaded** ✅")
            st.info("Model Evaluasi Kredit terkonfigurasi untuk memprediksi probabilitas gagal bayar.")
        else:
            st.warning("Status: **Model file tidak ditemukan** — mode simulasi tetap berjalan dengan rule-based scoring.")

    if total_sim:
        st.markdown("---")
        st.subheader("📈 Distribusi Keputusan (Semua Simulasi)")
        dist = log_df["decision"].value_counts().reset_index()
        dist.columns = ["decision", "count"]
        color_map = {"APPROVED": "#22c55e", "REVIEW": "#f59e0b", "BLOCKED": "#ef4444"}
        fig = px.bar(dist, x="decision", y="count", color="decision", color_discrete_map=color_map, text="count")
        fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada data simulasi. Jalankan analisis di modul **Fraud Simulator** atau **Credit Risk** untuk mengisi dashboard ini.")

# =============================================================================
# 10. FRAUD DETECTION SIMULATOR
# =============================================================================
elif menu == "🚨 Fraud Simulator":
    st.subheader("🚨 Fraud Detection Simulator")
    st.caption("Simulasikan data transaksi kartu untuk analisis potensi fraud secara real-time.")

    model = artifacts.get("fraud_model")
    features = artifacts.get("fraud_features", [])

    # ---- Preset skenario cepat ----
    for k, v in {
        "f_amount": 150000.0, "f_cust_tx": 80, "f_avg_tx": 120000.0,
        "f_m_rate": 0.01, "f_b_rate": 0.01, "f_freq": 1.0,
        "f_wnd": 0, "f_night": 0, "f_acc": "Savings",
    }.items():
        st.session_state.setdefault(k, v)

    preset = st.session_state.get("set_preset_flag")
    presets = {
        "normal": dict(f_amount=150000.0, f_cust_tx=80, f_avg_tx=120000.0, f_m_rate=0.01,
                       f_b_rate=0.01, f_freq=1.0, f_wnd=0, f_night=0, f_acc="Savings"),
        "medium": dict(f_amount=15000000.0, f_cust_tx=5, f_avg_tx=500000.0, f_m_rate=0.45,
                       f_b_rate=0.35, f_freq=6.0, f_wnd=1, f_night=0, f_acc="Checking"),
        "high": dict(f_amount=150000000.0, f_cust_tx=1, f_avg_tx=150000.0, f_m_rate=0.95,
                     f_b_rate=0.90, f_freq=25.0, f_wnd=1, f_night=1, f_acc="Checking"),
    }
    if preset in presets:
        for k, v in presets[preset].items():
            st.session_state[k] = v
        st.session_state.set_preset_flag = None

    st.markdown("**💡 Skenario Cepat:**")
    sc1, sc2, sc3 = st.columns(3)
    if sc1.button("🟢 Normal", use_container_width=True):
        st.session_state.set_preset_flag = "normal"; st.rerun()
    if sc2.button("🟡 Mencurigakan", use_container_width=True):
        st.session_state.set_preset_flag = "medium"; st.rerun()
    if sc3.button("🔴 Bahaya", use_container_width=True):
        st.session_state.set_preset_flag = "high"; st.rerun()

    def score_fraud(amount, customer_tx_count, avg_tx_amount, merchant_fraud_rate,
                     branch_fraud_rate, transaction_frequency, weekend_indicator,
                     night_indicator, account_type):
        amount_scaled = amount / 15000.0
        avg_tx_scaled = avg_tx_amount / 15000.0
        account_map = {"Savings": 0, "Checking": 1, "Premium": 2}

        prob = 0.0
        if model is not None and features:
            input_df = pd.DataFrame(0.0, index=[0], columns=features)
            mapping_dict = {
                "amount": amount_scaled, "transaction_amount": amount_scaled, "txn_amount": amount_scaled,
                "amount_raw": amount, "customer_tx_count": customer_tx_count, "tx_count": customer_tx_count,
                "avg_tx_amount": avg_tx_scaled, "avg_amount": avg_tx_scaled,
                "merchant_fraud_rate": merchant_fraud_rate, "merchant_risk": merchant_fraud_rate,
                "branch_fraud_rate": branch_fraud_rate, "branch_risk": branch_fraud_rate,
                "transaction_frequency": transaction_frequency, "freq": transaction_frequency,
                "weekend_transaction_indicator": weekend_indicator, "is_weekend": weekend_indicator,
                "night_transaction_indicator": night_indicator, "is_night": night_indicator,
                "account_type": account_map.get(account_type, 0),
            }
            for col in features:
                if col in mapping_dict:
                    input_df[col] = float(mapping_dict[col])
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(input_df)[0][1])
                else:
                    prob = float(model.predict(input_df)[0])
            except Exception:
                try:
                    X_vals = input_df.values
                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(X_vals)[0][1])
                    else:
                        prob = float(model.predict(X_vals)[0])
                except Exception:
                    prob = 0.0

        rule_score = (
            (merchant_fraud_rate * 0.4) + (branch_fraud_rate * 0.3)
            + (0.15 if night_indicator == 1 else 0.0)
            + (0.15 if (amount / max(avg_tx_amount, 1.0)) > 5.0 else 0.0)
        )
        prob = max(prob, rule_score)
        return min(max(prob, 0.0), 0.99)

    # ---------------------------------------------------------------
    # ⚡ REAL-TIME WHAT-IF SIMULATOR (tanpa tombol submit)
    # ---------------------------------------------------------------
    st.markdown("### ⚡ Real-Time What-If Simulator")
    st.caption("Geser slider di bawah — hasil risiko berubah **instan**, tanpa perlu menekan submit.")

    wcol1, wcol2 = st.columns([1.1, 1])
    with wcol1:
        wi_amount = st.slider("Nominal Transaksi (Rp)", 10_000, 300_000_000,
                               int(st.session_state.f_amount), step=10_000, format="Rp %d", key="wi_amount")
        wi_ratio = st.slider("Rasio Nominal vs Rata-rata Historis (x)", 0.5, 30.0, 1.5, step=0.1, key="wi_ratio")
        wi_merchant = st.slider("Merchant Risk Score", 0.0, 1.0, float(st.session_state.f_m_rate), key="wi_merchant")
        wi_branch = st.slider("Branch Risk Score", 0.0, 1.0, float(st.session_state.f_b_rate), key="wi_branch")
        wi_night = st.toggle("Transaksi Malam Hari (00–06)", value=bool(st.session_state.f_night), key="wi_night")
        wi_weekend = st.toggle("Transaksi Akhir Pekan", value=bool(st.session_state.f_wnd), key="wi_weekend")

    wi_avg = wi_amount / max(wi_ratio, 0.01)
    wi_prob = score_fraud(
        amount=wi_amount, customer_tx_count=int(st.session_state.f_cust_tx), avg_tx_amount=wi_avg,
        merchant_fraud_rate=wi_merchant, branch_fraud_rate=wi_branch,
        transaction_frequency=float(st.session_state.f_freq),
        weekend_indicator=int(wi_weekend), night_indicator=int(wi_night),
        account_type=st.session_state.f_acc,
    )
    wi_decision, wi_emoji = decision_from_prob(wi_prob)

    with wcol2:
        st.plotly_chart(create_gauge_chart(wi_prob, "Skor Risiko (Live)"), use_container_width=True)
        badge_color = {"APPROVED": "green", "REVIEW": "orange", "BLOCKED": "red"}[wi_decision]
        st.markdown(f"#### {wi_emoji} Status: **:{badge_color}[{wi_decision}]**")
        st.progress(min(wi_prob, 1.0), text=f"Skor risiko real-time: {wi_prob:.1%}")
        if st.button("➕ Simpan Skenario Ini ke Log", use_container_width=True, key="save_wi_fraud"):
            log_event(
                "Fraud (What-If)", wi_prob, wi_decision,
                amount=wi_amount, merchant_risk=wi_merchant, branch_risk=wi_branch,
                night=int(wi_night), weekend=int(wi_weekend),
            )

    extreme_risk_alert(wi_prob, label="Skenario What-If")

    st.markdown("---")

    # ---------------------------------------------------------------
    # Form analisis lengkap (tetap ada, untuk pencatatan resmi)
    # ---------------------------------------------------------------
    st.markdown("### 📋 Analisis Transaksi Lengkap (Form Resmi)")
    with st.form("fraud_form"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            amount = st.number_input("Jumlah Transaksi (Rp)", min_value=1000.0,
                                      value=float(st.session_state.f_amount), step=50000.0, format="%.0f")
            st.caption(f"Terbaca: **{format_rp(amount)}**")
            customer_tx_count = st.number_input("Total Riwayat Transaksi", min_value=1,
                                                 value=int(st.session_state.f_cust_tx))
            avg_tx_amount = st.number_input("Rata-rata Nominal (Rp)", min_value=1000.0,
                                             value=float(st.session_state.f_avg_tx), step=50000.0, format="%.0f")
            st.caption(f"Terbaca: **{format_rp(avg_tx_amount)}**")
        with col2:
            merchant_fraud_rate = st.slider("Tingkat Risk Merchant", 0.0, 1.0, float(st.session_state.f_m_rate))
            branch_fraud_rate = st.slider("Tingkat Risk Cabang", 0.0, 1.0, float(st.session_state.f_b_rate))
            transaction_frequency = st.number_input("Frekuensi Transaksi/Hari", min_value=0.1,
                                                      value=float(st.session_state.f_freq))
        with col3:
            weekend_indicator = st.selectbox("Akhir Pekan?", [0, 1], index=int(st.session_state.f_wnd),
                                              format_func=lambda x: "Ya" if x == 1 else "Tidak")
            night_indicator = st.selectbox("Malam Hari (00-06)?", [0, 1], index=int(st.session_state.f_night),
                                            format_func=lambda x: "Ya" if x == 1 else "Tidak")
            acc_idx = ["Savings", "Checking", "Premium"].index(st.session_state.f_acc)
            account_type = st.selectbox("Tipe Akun", ["Savings", "Checking", "Premium"], index=acc_idx)

        submit = st.form_submit_button("🔍 Jalankan Analisis Risiko", use_container_width=True)

    if submit:
        prob = score_fraud(amount, customer_tx_count, avg_tx_amount, merchant_fraud_rate,
                            branch_fraud_rate, transaction_frequency, weekend_indicator,
                            night_indicator, account_type)
        decision, emoji = decision_from_prob(prob)

        log_event("Fraud (Form)", prob, decision, amount=amount, merchant_risk=merchant_fraud_rate,
                   branch_risk=branch_fraud_rate, night=night_indicator, weekend=weekend_indicator,
                   account_type=account_type)

        st.markdown("---")
        st.subheader("Hasil Keputusan Transaksi:")
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.plotly_chart(create_gauge_chart(prob, "Skor Indikasi Fraud"), use_container_width=True)
        with res_col2:
            if decision == "BLOCKED":
                st.error("⛔ **STATUS: TRANSAKSI DITOLAK (DECLINED)**")
                st.metric(label="Keputusan Sistem", value="BLOCKED", delta="-HIGH RISK", delta_color="inverse")
                st.warning(f"**Transaksi Senilai {format_rp(amount)} Gagal Diinisiasi!**\n\n"
                           f"* Probabilitas: **{prob:.1%}**\n* Transaksi dibatalkan demi keamanan rekening.")
                st.toast("⚠️ Transaksi Berbahaya Diblokir Otomatis!", icon="⛔")
            elif decision == "REVIEW":
                st.warning("⚠️ **STATUS: DITAHAN / BUTUH VERIFIKASI (OTP)**")
                st.metric(label="Keputusan Sistem", value="CHALLENGE", delta="MEDIUM RISK", delta_color="off")
                st.info(f"**Transaksi Senilai {format_rp(amount)} Memerlukan OTP Kode:**\n\n"
                        f"* Probabilitas Risiko: **{prob:.1%}**\n* Minta nasabah verifikasi 6-digit.")
            else:
                st.success("✅ **STATUS: TRANSAKSI DISETUJUI (APPROVED)**")
                st.metric(label="Keputusan Sistem", value="SUCCESS", delta="LOW RISK")
                st.info(f"**Transaksi Senilai {format_rp(amount)} Berhasil!** Diproses ke core banking.")
                st.balloons()

        extreme_risk_alert(prob, label="Transaksi ini")

# =============================================================================
# 11. LOAN DEFAULT RISK PREDICTOR
# =============================================================================
elif menu == "💳 Credit Risk":
    st.subheader("💳 Loan Default Risk Predictor")
    st.caption("Evaluasi profil pemohon kredit pinjaman dan risiko gagal bayar (default).")

    model = artifacts.get("loan_model")
    features = artifacts.get("loan_features", [])

    def score_loan(loan_amount, customer_income, loan_duration, num_missed_payments,
                    account_balance, customer_age, support_ticket_count, loan_type):
        dti = loan_amount / max(customer_income, 1.0)
        loan_map = {"Personal": 0, "Auto": 1, "Mortgage": 2, "Business": 3}
        prob = 0.0
        if model is not None and features:
            input_df = pd.DataFrame(0.0, index=[0], columns=features)
            mapping_dict = {
                "loan_amount": loan_amount / 15000.0, "customer_income": customer_income / 15000.0,
                "loan_duration": loan_duration, "debt_to_income_ratio": dti,
                "num_missed_payments": num_missed_payments, "account_balance": account_balance / 15000.0,
                "customer_age": customer_age, "support_ticket_count": support_ticket_count,
                "loan_type": loan_map.get(loan_type, 0),
            }
            for col in features:
                if col in mapping_dict:
                    input_df[col] = float(mapping_dict[col])
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(input_df)[0][1])
                else:
                    prob = float(model.predict(input_df)[0])
            except Exception:
                try:
                    X_vals = input_df.values
                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(X_vals)[0][1])
                    else:
                        prob = float(model.predict(X_vals)[0])
                except Exception:
                    prob = 0.0
        credit_rule_score = (dti * 0.4) + (num_missed_payments * 0.15)
        prob = max(prob, credit_rule_score)
        return min(max(prob, 0.0), 0.99), dti

    # ---------------------------------------------------------------
    # ⚡ REAL-TIME WHAT-IF SIMULATOR
    # ---------------------------------------------------------------
    st.markdown("### ⚡ Real-Time What-If Simulator")
    st.caption("Geser DTI Ratio & jumlah keterlambatan bayar — hasil berubah instan.")

    wl1, wl2 = st.columns([1.1, 1])
    with wl1:
        wi_loan = st.slider("Pengajuan Pinjaman (Rp)", 1_000_000, 500_000_000, 50_000_000,
                             step=1_000_000, format="Rp %d", key="wi_loan")
        wi_income = st.slider("Pendapatan Tahunan (Rp)", 1_000_000, 500_000_000, 120_000_000,
                               step=1_000_000, format="Rp %d", key="wi_income")
        wi_missed = st.slider("Jumlah Terlambat Bayar", 0, 12, 0, key="wi_missed")
        wi_balance = st.slider("Saldo Tabungan (Rp)", 0, 200_000_000, 15_000_000,
                                step=1_000_000, format="Rp %d", key="wi_balance")
        wi_loan_type = st.selectbox("Jenis Pinjaman", ["Personal", "Auto", "Mortgage", "Business"], key="wi_ltype")

    wi_prob_loan, wi_dti = score_loan(
        loan_amount=wi_loan, customer_income=wi_income, loan_duration=36,
        num_missed_payments=wi_missed, account_balance=wi_balance,
        customer_age=32, support_ticket_count=0, loan_type=wi_loan_type,
    )
    wi_decision_loan, wi_emoji_loan = decision_from_prob(wi_prob_loan)

    with wl2:
        st.plotly_chart(create_gauge_chart(wi_prob_loan, "Skor Risiko Default (Live)"), use_container_width=True)
        badge_color = {"APPROVED": "green", "REVIEW": "orange", "BLOCKED": "red"}[wi_decision_loan]
        st.markdown(f"#### {wi_emoji_loan} Status: **:{badge_color}[{wi_decision_loan}]**")
        st.metric("DTI Ratio (Live)", f"{wi_dti:.1%}")
        st.progress(min(wi_prob_loan, 1.0), text=f"Skor risiko real-time: {wi_prob_loan:.1%}")
        if st.button("➕ Simpan Skenario Ini ke Log", use_container_width=True, key="save_wi_loan"):
            log_event("Kredit (What-If)", wi_prob_loan, wi_decision_loan,
                       loan_amount=wi_loan, income=wi_income, dti=round(wi_dti, 4),
                       missed_payments=wi_missed, loan_type=wi_loan_type)

    extreme_risk_alert(wi_prob_loan, label="Pengajuan ini")

    st.markdown("---")

    # ---------------------------------------------------------------
    # Form evaluasi lengkap
    # ---------------------------------------------------------------
    st.markdown("### 📋 Evaluasi Pengajuan Kredit (Form Resmi)")
    with st.form("loan_form"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            loan_amount = st.number_input("Pengajuan Pinjaman (Rp)", min_value=1000000.0,
                                           value=50000000.0, step=1000000.0, format="%.0f")
            st.caption(f"Terbaca: **{format_rp(loan_amount)}**")
            customer_income = st.number_input("Pendapatan Tahunan (Rp)", min_value=1000000.0,
                                               value=120000000.0, step=5000000.0, format="%.0f")
            st.caption(f"Terbaca: **{format_rp(customer_income)}**")
            loan_duration = st.number_input("Tenor Pinjaman (Bulan)", min_value=6, value=36)
        with col2:
            num_missed_payments = st.number_input("Jumlah Terlambat Bayar", min_value=0, value=0)
            account_balance = st.number_input("Saldo Tabungan Saat Ini (Rp)", min_value=0.0,
                                               value=15000000.0, step=1000000.0, format="%.0f")
            st.caption(f"Terbaca: **{format_rp(account_balance)}**")
            loan_type = st.selectbox("Jenis Pinjaman", ["Personal", "Auto", "Mortgage", "Business"])
        with col3:
            customer_age = st.number_input("Usia Pemohon", min_value=18, value=32)
            support_ticket_count = st.number_input("Jumlah Komplain Nasabah", min_value=0, value=0)

        submit = st.form_submit_button("📊 Evaluasi Risiko Pengajuan", use_container_width=True)

    if submit:
        prob, dti = score_loan(loan_amount, customer_income, loan_duration, num_missed_payments,
                                account_balance, customer_age, support_ticket_count, loan_type)
        decision, emoji = decision_from_prob(prob)

        log_event("Kredit (Form)", prob, decision, loan_amount=loan_amount, income=customer_income,
                   dti=round(dti, 4), missed_payments=num_missed_payments, loan_type=loan_type)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rasio DTI", f"{dti:.2%}")
        c2.metric("Probabilitas Gagal Bayar", f"{prob:.2%}")
        if decision == "BLOCKED":
            c3.metric("Rekomendasi", "DITOLAK ❌", delta="-HIGH RISK", delta_color="inverse")
        elif decision == "REVIEW":
            c3.metric("Rekomendasi", "REVIEW MANUAL 🟡", delta="MEDIUM RISK", delta_color="off")
        else:
            c3.metric("Rekomendasi", "DISETUJUI ✅", delta="LOW RISK")
            st.balloons()

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.plotly_chart(create_gauge_chart(prob, "Skor Risiko Default Kredit"), use_container_width=True)
        with res_col2:
            st.subheader("📌 Analisis Profil Finansial")
            fig_bar = go.Figure(go.Bar(
                x=[loan_amount, customer_income, account_balance],
                y=['Pinjaman', 'Pendapatan/Thn', 'Saldo Tabungan'],
                orientation='h',
                text=[format_rp(loan_amount), format_rp(customer_income), format_rp(account_balance)],
                textposition='auto', marker_color=['#ef5350', '#26a69a', '#42a5f5'],
            ))
            fig_bar.update_layout(height=220, margin=dict(l=15, r=15, t=15, b=15),
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)

        extreme_risk_alert(prob, label="Pengajuan ini")

# =============================================================================
# 12. LOG & ANALYTICS (Tabs: Ringkasan | Feature Importance | Raw Data)
# =============================================================================
elif menu == "📜 Log & Analytics":
    st.subheader("📜 Histori Log Transaksi Simulasi")
    st.caption("Semua hasil dari Real-Time Simulator maupun Form Resmi tercatat di sini.")

    log_df = pd.DataFrame(st.session_state.simulation_log)

    tab1, tab2, tab3 = st.tabs(
        ["📊 Ringkasan Status & Metrics", "🔬 Feature Importance", "🗂️ Simulation Log Data Raw"]
    )

    # ---- TAB 1: Ringkasan ----
    with tab1:
        if log_df.empty:
            st.info("Belum ada data. Jalankan simulasi di modul Fraud Simulator / Credit Risk lalu klik **'Simpan ke Log'** atau submit form.")
        else:
            total = len(log_df)
            n_approved = int((log_df["decision"] == "APPROVED").sum())
            n_review = int((log_df["decision"] == "REVIEW").sum())
            n_blocked = int((log_df["decision"] == "BLOCKED").sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Entri Log", total)
            k2.metric("Approved", n_approved, delta=f"{n_approved/total:.0%}")
            k3.metric("Review", n_review, delta=f"{n_review/total:.0%}", delta_color="off")
            k4.metric("Blocked", n_blocked, delta=f"{n_blocked/total:.0%}", delta_color="inverse")

            cc1, cc2 = st.columns(2)
            with cc1:
                dist = log_df["decision"].value_counts().reset_index()
                dist.columns = ["decision", "count"]
                color_map = {"APPROVED": "#22c55e", "REVIEW": "#f59e0b", "BLOCKED": "#ef4444"}
                fig = px.pie(dist, names="decision", values="count", hole=0.5,
                             color="decision", color_discrete_map=color_map)
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with cc2:
                trend = log_df.copy()
                trend["idx"] = range(1, len(trend) + 1)
                fig2 = px.line(trend, x="idx", y="probability", markers=True,
                                color="module" if "module" in trend else None)
                fig2.add_hline(y=get_thresholds()[0], line_dash="dot", line_color="#22c55e")
                fig2.add_hline(y=get_thresholds()[1], line_dash="dot", line_color="#ef4444")
                fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                    xaxis_title="Urutan Simulasi", yaxis_title="Probabilitas Risiko")
                st.plotly_chart(fig2, use_container_width=True)

    # ---- TAB 2: Feature Importance ----
    with tab2:
        fi_col1, fi_col2 = st.columns(2)
        with fi_col1:
            st.markdown("##### 🚨 Fraud Model")
            fraud_fallback = {
                "merchant_fraud_rate": 0.40, "branch_fraud_rate": 0.30,
                "night_transaction_indicator": 0.15, "amount_vs_avg_spike": 0.15,
            }
            df_fi_fraud, src_fraud = build_feature_importance(
                artifacts.get("fraud_model"), artifacts.get("fraud_features", []),
                fraud_fallback, "Fraud Model",
            )
            st.caption(f"Sumber: {src_fraud}")
            fig = px.bar(df_fi_fraud.sort_values("importance"), x="importance", y="feature", orientation="h")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with fi_col2:
            st.markdown("##### 💳 Credit Risk Model")
            loan_fallback = {"debt_to_income_ratio": 0.40, "num_missed_payments": 0.15}
            df_fi_loan, src_loan = build_feature_importance(
                artifacts.get("loan_model"), artifacts.get("loan_features", []),
                loan_fallback, "Credit Risk Model",
            )
            st.caption(f"Sumber: {src_loan}")
            fig = px.bar(df_fi_loan.sort_values("importance"), x="importance", y="feature", orientation="h")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 3: Raw log dengan filter, search, export ----
    with tab3:
        if log_df.empty:
            st.info("Log masih kosong.")
        else:
            fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
            with fcol1:
                module_filter = st.multiselect("Filter Modul", sorted(log_df["module"].unique()),
                                                default=sorted(log_df["module"].unique()))
            with fcol2:
                decision_filter = st.multiselect("Filter Keputusan", sorted(log_df["decision"].unique()),
                                                  default=sorted(log_df["decision"].unique()))
            with fcol3:
                search_term = st.text_input("🔎 Cari di seluruh kolom log", "")

            filtered = log_df[
                log_df["module"].isin(module_filter) & log_df["decision"].isin(decision_filter)
            ]
            if search_term:
                mask = filtered.apply(
                    lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1
                )
                filtered = filtered[mask]

            st.dataframe(filtered, use_container_width=True, hide_index=True)

            dcol1, dcol2, dcol3 = st.columns([1, 1, 3])
            with dcol1:
                csv_bytes = filtered.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Log CSV", data=csv_bytes,
                                    file_name="bankguard_simulation_log.csv", mime="text/csv",
                                    use_container_width=True)
            with dcol2:
                json_bytes = json.dumps(filtered.to_dict(orient="records"), indent=2).encode("utf-8")
                st.download_button("⬇️ Download Log JSON", data=json_bytes,
                                    file_name="bankguard_simulation_log.json", mime="application/json",
                                    use_container_width=True)
            with dcol3:
                if st.button("🗑️ Bersihkan Log", use_container_width=True):
                    st.session_state.simulation_log = []
                    st.toast("Log berhasil dikosongkan.", icon="🗑️")
                    st.rerun()
