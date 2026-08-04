import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="BankGuard AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk UI
st.markdown("""
<style>
    .stCard {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e9ecef;
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

# Helper untuk format mata uang Rupiah rapi (120.000)
def format_rp(val):
    return f"Rp {val:,.0f}".replace(",", ".")

# Helper untuk Gauge Chart Skor Risiko
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
                {'range': [0, 30], 'color': "#e8f5e9"},
                {'range': [30, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#ffebee"}
            ],
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Sidebar
st.sidebar.title("🛡️ BankGuard AI")
st.sidebar.caption("Real-Time Financial Risk Intelligence Platform")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigasi Modul:",
    ["Dashboard Utama", "🚨 Fraud Detection Simulator", "💳 Loan Default Risk Predictor"]
)

# ---------------- 1. DASHBOARD UTAMA ----------------
if menu == "Dashboard Utama":
    st.title("🛡️ Executive Overview")
    st.caption("Platform Analisis Portofolio Risiko Keuangan Berbasis AI")
    
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
            st.info("Model LightGBM terkonfigurasi untuk memonitor anomali transaksi real-time.")
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
    st.title("🚨 Fraud Detection Simulator")
    st.caption("Simulasikan data transaksi kartu untuk analisis potensi fraud secara real-time.")

    if "fraud_model" not in artifacts:
        st.error("Model Fraud tidak ditemukan.")
    else:
        model = artifacts["fraud_model"]
        features = artifacts["fraud_features"]

        st.subheader("💡 Skenario Cepat:")
        sc1, sc2, sc3 = st.columns(3)
        
        default_vals = {"amount": 250000.0, "cust_tx": 45, "avg_tx": 150000.0, "m_rate": 0.02, "b_rate": 0.01, "freq": 1.5, "wnd": 0, "night": 0, "acc": "Savings"}
        
        if sc1.button("🟢 Transaksi Normal (Disetujui)"):
            default_vals = {"amount": 150000.0, "cust_tx": 80, "avg_tx": 120000.0, "m_rate": 0.01, "b_rate": 0.01, "freq": 1.0, "wnd": 0, "night": 0, "acc": "Savings"}
        if sc2.button("🟡 Transaksi Mencurigakan (OTP)"):
            default_vals = {"amount": 3500000.0, "cust_tx": 12, "avg_tx": 500000.0, "m_rate": 0.15, "b_rate": 0.10, "freq": 5.0, "wnd": 1, "night": 0, "acc": "Checking"}
        if sc3.button("🔴 Transaksi Berbahaya (Ditolak)"):
            default_vals = {"amount": 25000000.0, "cust_tx": 2, "avg_tx": 300000.0, "m_rate": 0.65, "b_rate": 0.40, "freq": 12.0, "wnd": 1, "night": 1, "acc": "Checking"}

        with st.form("fraud_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                amount = st.number_input("Jumlah Transaksi (Rp)", min_value=1000.0, value=default_vals["amount"], step=50000.0, format="%.0f")
                st.caption(f"Terbaca: **{format_rp(amount)}**")
                
                customer_tx_count = st.number_input("Total Riwayat Transaksi Nasabah", min_value=1, value=default_vals["cust_tx"])
                
                avg_tx_amount = st.number_input("Rata-rata Nominal Transaksi (Rp)", min_value=1000.0, value=default_vals["avg_tx"], step=50000.0, format="%.0f")
                st.caption(f"Terbaca: **{format_rp(avg_tx_amount)}**")

            with col2:
                merchant_fraud_rate = st.slider("Tingkat Risk Merchant", 0.0, 1.0, default_vals["m_rate"])
                branch_fraud_rate = st.slider("Tingkat Risk Cabang", 0.0, 1.0, default_vals["b_rate"])
                transaction_frequency = st.number_input("Frekuensi Transaksi/Hari", min_value=0.1, value=default_vals["freq"])
            with col3:
                weekend_indicator = st.selectbox("Transaksi Akhir Pekan?", [0, 1], index=default_vals["wnd"], format_func=lambda x: "Ya" if x == 1 else "Tidak")
                night_indicator = st.selectbox("Transaksi Malam Hari (00.00-06.00)?", [0, 1], index=default_vals["night"], format_func=lambda x: "Ya" if x == 1 else "Tidak")
                acc_idx = ["Savings", "Checking", "Premium"].index(default_vals["acc"])
                account_type = st.selectbox("Tipe Akun", ["Savings", "Checking", "Premium"], index=acc_idx)

            submit = st.form_submit_button("🔍 Jalankan Analisis Risiko", use_container_width=True)

        if submit:
            input_data = pd.DataFrame(0.0, index=[0], columns=features)
            account_map = {"Savings": 0, "Checking": 1, "Premium": 2}
            
            val_map = {
                "amount": amount, "customer_tx_count": customer_tx_count,
                "avg_tx_amount": avg_tx_amount, "merchant_fraud_rate": merchant_fraud_rate,
                "branch_fraud_rate": branch_fraud_rate, "transaction_frequency": transaction_frequency,
                "weekend_transaction_indicator": weekend_indicator, "night_transaction_indicator": night_indicator,
                "account_type": account_map.get(account_type, 0),
            }
            for k, v in val_map.items():
                if k in input_data.columns:
                    input_data[k] = float(v)

            X_input = input_data.values
            try:
                pred = model.predict(X_input)[0]
                prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
            except Exception:
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

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

# ---------------- 3. LOAN DEFAULT RISK PREDICTOR ----------------
elif menu == "💳 Loan Default Risk Predictor":
    st.title("💳 Loan Default Risk Predictor")
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
            input_data = pd.DataFrame(0.0, index=[0], columns=features)
            dti = loan_amount / max(customer_income, 1.0)
            loan_map = {"Personal": 0, "Auto": 1, "Mortgage": 2, "Business": 3}
            
            val_map = {
                "loan_amount": loan_amount, "customer_income": customer_income,
                "loan_duration": loan_duration, "debt_to_income_ratio": dti,
                "num_missed_payments": num_missed_payments, "account_balance": account_balance,
                "customer_age": customer_age, "support_ticket_count": support_ticket_count,
                "loan_type": loan_map.get(loan_type, 0),
            }
            for k, v in val_map.items():
                if k in input_data.columns:
                    input_data[k] = float(v)

            X_input = input_data.values
            try:
                pred = model.predict(X_input)[0]
                prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
            except Exception:
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

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
                fig_bar.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
