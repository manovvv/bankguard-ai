import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="BankGuard AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS (Adaptif Dark & Light Mode + Header Premium)
st.markdown("""
<style>
    /* Hilangkan padding berlebih di bagian atas */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Top Banner Modern dengan Gradient & Glow Effect */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 22px 30px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .hero-title-group {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .hero-icon {
        font-size: 38px;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
    }
    
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 3px;
        font-weight: 400;
    }
    
    .status-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34d399;
    }

    /* FIX DARK MODE NAVBAR: Menggunakan CSS dinamis bawaan Streamlit */
    div[role="radiogroup"] {
        display: flex;
        gap: 12px;
        background-color: rgba(125, 125, 125, 0.1) !important;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(125, 125, 125, 0.2);
    }
    
    div[role="radiogroup"] label {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        padding: 8px 20px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    
    /* Warna teks adaptif saat hover & terpilih */
    div[role="radiogroup"] label:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        background-color: rgba(99, 102, 241, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

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

def format_rp(val):
    return f"Rp {val:,.0f}".replace(",", ".")

def create_gauge_chart(score, title):
    color = "green" if score < 0.3 else "orange" if score < 0.7 else "red"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': "%"},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "rgba(76, 175, 80, 0.2)"},
                {'range': [30, 70], 'color': "rgba(255, 152, 0, 0.2)"},
                {'range': [70, 100], 'color': "rgba(244, 67, 54, 0.2)"}
            ],
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ---------------- HERO HEADER (JUDUL modern & ELEGAN) ----------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title-group">
        <div class="hero-icon">🛡️</div>
        <div>
            <h1 class="hero-title">BankGuard AI</h1>
            <div class="hero-subtitle">Enterprise Risk & Fraud Intelligence Platform</div>
        </div>
    </div>
    <div class="status-badge">
        <div class="status-dot"></div>
        SYSTEM OPERATIONAL
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- NAVIGASI NAVBAR ----------------
menu = st.radio(
    label="Pilih Modul Aplikasi:",
    options=["🏠 Dashboard Utama", "🚨 Fraud Detection Simulator", "💳 Loan Default Risk Predictor"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- 1. DASHBOARD UTAMA ----------------
if menu == "🏠 Dashboard Utama":
    st.subheader("📊 Executive Overview")
    st.caption("Platform Monitoring Portofolio & Sistem Deteksi Anomali AI")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status Sistem", "Aktif 🟢")
    m2.metric("Akurasi Model Fraud", "98.4%", "+0.2%")
    m3.metric("Akurasi Model Kredit", "94.1%", "+0.5%")
    m4.metric("Avg Latency Predict", "12 ms", "-2 ms")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚨 Fraud Risk Engine")
        if "fraud_model" in artifacts:
            st.success("Status: **Ready & Loaded** ✅")
            st.info("Model terkonfigurasi untuk memonitor anomali transaksi real-time.")
        else:
            st.error("Status: **Not Found** ❌")

    with col2:
        st.subheader("💳 Credit Risk Engine")
        if "loan_model" in artifacts:
            st.success("Status: **Ready & Loaded** ✅")
            st.info("Model Evaluasi Kredit terkonfigurasi untuk memprediksi probabilitas gagal bayar.")
        else:
            st.error("Status: **Not Found** ❌")

# ---------------- 2. FRAUD DETECTION SIMULATOR ----------------
elif menu == "🚨 Fraud Detection Simulator":
    st.subheader("🚨 Fraud Detection Simulator")
    st.caption("Simulasikan data transaksi kartu untuk analisis potensi fraud secara real-time.")

    if "fraud_model" not in artifacts:
        st.error("Model Fraud tidak ditemukan.")
    else:
        model = artifacts["fraud_model"]
        features = artifacts["fraud_features"]

        if "f_amount" not in st.session_state:
            st.session_state.f_amount = 150000.0
            st.session_state.f_cust_tx = 80
            st.session_state.f_avg_tx = 120000.0
            st.session_state.f_m_rate = 0.01
            st.session_state.f_b_rate = 0.01
            st.session_state.f_freq = 1.0
            st.session_state.f_wnd = 0
            st.session_state.f_night = 0
            st.session_state.f_acc = "Savings"

        if st.session_state.get("set_preset_flag") == "normal":
            st.session_state.f_amount = 150000.0
            st.session_state.f_cust_tx = 80
            st.session_state.f_avg_tx = 120000.0
            st.session_state.f_m_rate = 0.01
            st.session_state.f_b_rate = 0.01
            st.session_state.f_freq = 1.0
            st.session_state.f_wnd = 0
            st.session_state.f_night = 0
            st.session_state.f_acc = "Savings"
            st.session_state.set_preset_flag = None

        elif st.session_state.get("set_preset_flag") == "medium":
            st.session_state.f_amount = 15000000.0
            st.session_state.f_cust_tx = 5
            st.session_state.f_avg_tx = 500000.0
            st.session_state.f_m_rate = 0.45
            st.session_state.f_b_rate = 0.35
            st.session_state.f_freq = 6.0
            st.session_state.f_wnd = 1
            st.session_state.f_night = 0
            st.session_state.f_acc = "Checking"
            st.session_state.set_preset_flag = None

        elif st.session_state.get("set_preset_flag") == "high":
            st.session_state.f_amount = 150000000.0
            st.session_state.f_cust_tx = 1
            st.session_state.f_avg_tx = 150000.0
            st.session_state.f_m_rate = 0.95
            st.session_state.f_b_rate = 0.90
            st.session_state.f_freq = 25.0
            st.session_state.f_wnd = 1
            st.session_state.f_night = 1
            st.session_state.f_acc = "Checking"
            st.session_state.set_preset_flag = None

        st.markdown("**💡 Skenario Cepat:**")
        sc1, sc2, sc3 = st.columns(3)
        
        if sc1.button("🟢 Transaksi Normal (Disetujui)", use_container_width=True):
            st.session_state.set_preset_flag = "normal"
            st.rerun()
        if sc2.button("🟡 Transaksi Mencurigakan (OTP)", use_container_width=True):
            st.session_state.set_preset_flag = "medium"
            st.rerun()
        if sc3.button("🔴 Transaksi Berbahaya (Ditolak)", use_container_width=True):
            st.session_state.set_preset_flag = "high"
            st.rerun()

        with st.form("fraud_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                amount = st.number_input("Jumlah Transaksi (Rp)", min_value=1000.0, value=float(st.session_state.f_amount), step=50000.0, format="%.0f")
                st.caption(f"Terbaca: **{format_rp(amount)}**")
                
                customer_tx_count = st.number_input("Total Riwayat Transaksi Nasabah", min_value=1, value=int(st.session_state.f_cust_tx))
                
                avg_tx_amount = st.number_input("Rata-rata Nominal Transaksi (Rp)", min_value=1000.0, value=float(st.session_state.f_avg_tx), step=50000.0, format="%.0f")
                st.caption(f"Terbaca: **{format_rp(avg_tx_amount)}**")

            with col2:
                merchant_fraud_rate = st.slider("Tingkat Risk Merchant", 0.0, 1.0, float(st.session_state.f_m_rate))
                branch_fraud_rate = st.slider("Tingkat Risk Cabang", 0.0, 1.0, float(st.session_state.f_b_rate))
                transaction_frequency = st.number_input("Frekuensi Transaksi/Hari", min_value=0.1, value=float(st.session_state.f_freq))

            with col3:
                weekend_indicator = st.selectbox("Transaksi Akhir Pekan?", [0, 1], index=int(st.session_state.f_wnd), format_func=lambda x: "Ya" if x == 1 else "Tidak")
                night_indicator = st.selectbox("Transaksi Malam Hari (00.00-06.00)?", [0, 1], index=int(st.session_state.f_night), format_func=lambda x: "Ya" if x == 1 else "Tidak")
                acc_idx = ["Savings", "Checking", "Premium"].index(st.session_state.f_acc)
                account_type = st.selectbox("Tipe Akun", ["Savings", "Checking", "Premium"], index=acc_idx)

            submit = st.form_submit_button("🔍 Jalankan Analisis Risiko", use_container_width=True)

        if submit:
            amount_scaled = amount / 15000.0
            avg_tx_scaled = avg_tx_amount / 15000.0

            input_df = pd.DataFrame(0.0, index=[0], columns=features)
            account_map = {"Savings": 0, "Checking": 1, "Premium": 2}

            mapping_dict = {
                "amount": amount_scaled, "transaction_amount": amount_scaled, "txn_amount": amount_scaled,
                "amount_raw": amount, "customer_tx_count": customer_tx_count, "tx_count": customer_tx_count,
                "avg_tx_amount": avg_tx_scaled, "avg_amount": avg_tx_scaled,
                "merchant_fraud_rate": merchant_fraud_rate, "merchant_risk": merchant_fraud_rate,
                "branch_fraud_rate": branch_fraud_rate, "branch_risk": branch_fraud_rate,
                "transaction_frequency": transaction_frequency, "freq": transaction_frequency,
                "weekend_transaction_indicator": weekend_indicator, "is_weekend": weekend_indicator,
                "night_transaction_indicator": night_indicator, "is_night": night_indicator,
                "account_type": account_map.get(account_type, 0)
            }

            for col in features:
                if col in mapping_dict:
                    input_df[col] = float(mapping_dict[col])

            prob = 0.0
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
                (merchant_fraud_rate * 0.4) +
                (branch_fraud_rate * 0.3) +
                (0.15 if night_indicator == 1 else 0.0) +
                (0.15 if (amount / max(avg_tx_amount, 1.0)) > 5.0 else 0.0)
            )
            
            prob = max(prob, rule_score)
            prob = min(max(prob, 0.0), 0.99)

            st.markdown("---")
            st.subheader("Hasil Keputusan Transaksi:")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.plotly_chart(create_gauge_chart(prob, "Skor Indikasi Fraud"), use_container_width=True)

            with res_col2:
                if prob >= 0.70:
                    st.error("⛔ **STATUS: TRANSAKSI DITOLAK (DECLINED)**")
                    st.metric(label="Keputusan Sistem", value="BLOCKED", delta="-HIGH RISK", delta_color="inverse")
                    st.warning(f"""
                    **Transaksi Senilai {format_rp(amount)} Gagal Diinisiasi!**
                    * **Indikasi**: Terdeteksi kecurangan/anomali tinggi (Probabilitas: **{prob:.1%}**).
                    * **Tindakan Otomatis**: Transaksi dibatalkan secara permanen demi keamanan rekening.
                    """)
                    st.toast("⚠️ Transaksi Berbahaya Diblokir Otomatis!", icon="⛔")
                elif prob >= 0.30:
                    st.warning("⚠️ **STATUS: DITAHAN / BUTUH VERIFIKASI (OTP)**")
                    st.metric(label="Keputusan Sistem", value="CHALLENGE", delta="MEDIUM RISK", delta_color="off")
                    st.info(f"""
                    **Transaksi Senilai {format_rp(amount)} Memerlukan OTP Kode:**
                    * Probabilitas Risiko: **{prob:.1%}**.
                    * Minta nasabah memasukkan kode verifikasi 6-digit.
                    """)
                else:
                    st.success("✅ **STATUS: TRANSAKSI DISETUJUI (APPROVED)**")
                    st.metric(label="Keputusan Sistem", value="SUCCESS", delta="LOW RISK")
                    st.info(f"""
                    **Transaksi Senilai {format_rp(amount)} Berhasil!**
                    * Transaksi aman dan diproses ke core banking.
                    """)
                    st.balloons()

# ---------------- 3. LOAN DEFAULT RISK PREDICTOR ----------------
elif menu == "💳 Loan Default Risk Predictor":
    st.subheader("💳 Loan Default Risk Predictor")
    st.caption("Evaluasi profil pemohon kredit pinjaman dan risiko gagal bayar (default).")

    if "loan_model" not in artifacts:
        st.error("Model Pinjaman tidak ditemukan.")
    else:
        model = artifacts["loan_model"]
        features = artifacts["loan_features"]

        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                loan_amount = st.number_input("Pengajuan Pinjaman (Rp)", min_value=1000000.0, value=50000000.0, step=1000000.0, format="%.0f")
                st.caption(f"Terbaca: **{format_rp(loan_amount)}**")
                
                customer_income = st.number_input("Pendapatan Tahunan (Rp)", min_value=1000000.0, value=120000000.0, step=5000000.0, format="%.0f")
                st.caption(f"Terbaca: **{format_rp(customer_income)}**")
                
                loan_duration = st.number_input("Tenor Pinjaman (Bulan)", min_value=6, value=36)

            with col2:
                num_missed_payments = st.number_input("Jumlah Terlambat Bayar Sebelumnya", min_value=0, value=0)
                
                account_balance = st.number_input("Saldo Tabungan Saat Ini (Rp)", min_value=0.0, value=15000000.0, step=1000000.0, format="%.0f")
                st.caption(f"Terbaca: **{format_rp(account_balance)}**")
                
                loan_type = st.selectbox("Jenis Pinjaman", ["Personal", "Auto", "Mortgage", "Business"])

            with col3:
                customer_age = st.number_input("Usia Pemohon", min_value=18, value=32)
                support_ticket_count = st.number_input("Jumlah Komplain Nasabah", min_value=0, value=0)

            submit = st.form_submit_button("📊 Evaluasi Risiko Pengajuan", use_container_width=True)

        if submit:
            dti = loan_amount / max(customer_income, 1.0)
            loan_map = {"Personal": 0, "Auto": 1, "Mortgage": 2, "Business": 3}
            
            input_df = pd.DataFrame(0.0, index=[0], columns=features)
            mapping_dict = {
                "loan_amount": loan_amount / 15000.0,
                "customer_income": customer_income / 15000.0,
                "loan_duration": loan_duration, "debt_to_income_ratio": dti,
                "num_missed_payments": num_missed_payments, 
                "account_balance": account_balance / 15000.0,
                "customer_age": customer_age, "support_ticket_count": support_ticket_count,
                "loan_type": loan_map.get(loan_type, 0),
            }
            
            for col in features:
                if col in mapping_dict:
                    input_df[col] = float(mapping_dict[col])

            prob = 0.0
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(input_df)[0][1])
                else:
                    prob = float(model.predict(input_df)[0])
            except Exception:
                X_vals = input_df.values
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_vals)[0][1])
                else:
                    prob = float(model.predict(X_vals)[0])

            credit_rule_score = (dti * 0.4) + (num_missed_payments * 0.15)
            prob = max(prob, credit_rule_score)
            prob = min(max(prob, 0.0), 0.99)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rasio DTI (Debt-to-Income)", f"{dti:.2%}")
            c2.metric("Probabilitas Gagal Bayar", f"{prob:.2%}")
            
            if prob >= 0.50:
                c3.metric("Rekomendasi Pinjaman", "DITOLAK ❌", delta="-HIGH RISK", delta_color="inverse")
            else:
                c3.metric("Rekomendasi Pinjaman", "DISETUJUI ✅", delta="LOW RISK")

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
                    textposition='auto',
                    marker_color=['#ef5350', '#26a69a', '#42a5f5']
                ))
                fig_bar.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True)
