import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BankGuard AI Platform",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path(".")
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def load_artifacts():
    artifacts = {}
    
    fraud_model_path = MODELS_DIR / "fraud_model.pkl"
    fraud_feat_path = MODELS_DIR / "fraud_features.pkl"
    loan_model_path = MODELS_DIR / "loan_model.pkl"
    loan_feat_path = MODELS_DIR / "loan_features.pkl"

    if fraud_model_path.exists() and fraud_feat_path.exists():
        with open(fraud_model_path, "rb") as f:
            artifacts["fraud_model"] = pickle.load(f)
        with open(fraud_feat_path, "rb") as f:
            artifacts["fraud_features"] = pickle.load(f)

    if loan_model_path.exists() and loan_feat_path.exists():
        with open(loan_model_path, "rb") as f:
            artifacts["loan_model"] = pickle.load(f)
        with open(loan_feat_path, "rb") as f:
            artifacts["loan_features"] = pickle.load(f)

    return artifacts


artifacts = load_artifacts()

st.sidebar.title("🛡️ BankGuard AI")
st.sidebar.caption("Real-Time Financial Risk Intelligence")

menu = st.sidebar.radio(
    "Navigasi Modul:",
    ["Overview", "🚨 Fraud Detection Simulator", "💳 Loan Default Risk Predictor"]
)

if menu == "Overview":
    st.title("🛡️ Welcome to BankGuard AI Platform")
    st.markdown("""
    **BankGuard AI** adalah platform analitik risiko keuangan berbasis Machine Learning yang dirancang untuk:
    - **Deteksi Dini Fraud Transaksi**: Mengidentifikasi transaksi kartu mencurigakan menggunakan LightGBM.
    - **Evaluasi Risiko Kredit Pinjaman**: Memprediksi probabilitas gagal bayar (*default*) pada kredit nasabah.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 🚨 Fraud Model Status\n" + ("Ready ✅" if "fraud_model" in artifacts else "Not Found ❌"))
    with col2:
        st.success("### 💳 Loan Risk Model Status\n" + ("Ready ✅" if "loan_model" in artifacts else "Not Found ❌"))

elif menu == "🚨 Fraud Detection Simulator":
    st.title("🚨 Fraud Detection Simulator")
    st.markdown("Masukkan rincian transaksi kartu untuk memprediksi potensi *fraud* secara real-time.")

    if "fraud_model" not in artifacts:
        st.error("Model Fraud (`fraud_model.pkl`) tidak ditemukan di direktori utama.")
    else:
        model = artifacts["fraud_model"]
        features = artifacts["fraud_features"]

        with st.form("fraud_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                amount = st.number_input("Jumlah Transaksi ($)", min_value=1.0, value=250.0, step=10.0)
                customer_tx_count = st.number_input("Total Transaksi Nasabah", min_value=1, value=45)
                avg_tx_amount = st.number_input("Rata-rata Transaksi ($)", min_value=1.0, value=85.0)

            with col2:
                merchant_fraud_rate = st.slider("Tingkat Risk Merchant", 0.0, 1.0, 0.02)
                branch_fraud_rate = st.slider("Tingkat Risk Cabang", 0.0, 1.0, 0.01)
                transaction_frequency = st.number_input("Frekuensi Transaksi/Hari", min_value=0.1, value=1.5)

            with col3:
                weekend_indicator = st.selectbox("Transaksi Akhir Pekan?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
                night_indicator = st.selectbox("Transaksi Malam Hari (00.00-06.00)?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
                account_type = st.selectbox("Tipe Akun", ["Savings", "Checking", "Premium"])

            submit = st.form_submit_button("🔍 Prediksi Fraud")

        if submit:
            input_data = pd.DataFrame(0, index=[0], columns=features)
            
            val_map = {
                "amount": amount,
                "customer_tx_count": customer_tx_count,
                "avg_tx_amount": avg_tx_amount,
                "merchant_fraud_rate": merchant_fraud_rate,
                "branch_fraud_rate": branch_fraud_rate,
                "transaction_frequency": transaction_frequency,
                "weekend_transaction_indicator": weekend_indicator,
                "night_transaction_indicator": night_indicator,
                "account_type": account_type,
            }
            
            for k, v in val_map.items():
                if k in input_data.columns:
                    input_data[k] = v

            if "account_type" in input_data.columns:
                input_data["account_type"] = input_data["account_type"].astype("category")

            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

            st.markdown("---")
            st.subheader("Hasil Analisis Transaksi:")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if pred == 1 or prob >= 0.5:
                    st.error(f"🚨 **HIGH RISK: INDIKASI FRAUD DETECTED**\n\nProbabilitas Fraud: **{prob:.2%}**")
                else:
                    st.success(f"✅ **LOW RISK: TRANSAKSI NORMAL**\n\nProbabilitas Fraud: **{prob:.2%}**")

            with res_col2:
                st.metric("Skor Probabilitas Risiko Fraud", f"{prob:.4f}")

elif menu == "💳 Loan Default Risk Predictor":
    st.title("💳 Loan Default Risk Predictor")
    st.markdown("Evaluasi profil nasabah untuk memprediksi potensi gagal bayar pinjaman (*Credit Default*).")

    if "loan_model" not in artifacts:
        st.error("Model Risiko Pinjaman (`loan_model.pkl`) tidak ditemukan di direktori utama.")
    else:
        model = artifacts["loan_model"]
        features = artifacts["loan_features"]

        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=500.0, value=15000.0, step=500.0)
                customer_income = st.number_input("Pendapatan Tahunan ($)", min_value=1000.0, value=55000.0, step=1000.0)
                loan_duration = st.number_input("Durasi Pinjaman (Bulan)", min_value=6, value=36)

            with col2:
                num_missed_payments = st.number_input("Jumlah Terlambat Bayar", min_value=0, value=0)
                account_balance = st.number_input("Saldo Tabungan Saat Ini ($)", min_value=0.0, value=4500.0)
                loan_type = st.selectbox("Jenis Pinjaman", ["Personal", "Auto", "Mortgage", "Business"])

            with col3:
                customer_age = st.number_input("Usia Nasabah", min_value=18, value=35)
                support_ticket_count = st.number_input("Jumlah Tiket Komplain", min_value=0, value=1)

            submit = st.form_submit_button("📊 Evaluasi Risiko Pinjaman")

        if submit:
            input_data = pd.DataFrame(0, index=[0], columns=features)
            
            dti = loan_amount / max(customer_income, 1.0)
            
            val_map = {
                "loan_amount": loan_amount,
                "customer_income": customer_income,
                "loan_duration": loan_duration,
                "debt_to_income_ratio": dti,
                "num_missed_payments": num_missed_payments,
                "account_balance": account_balance,
                "customer_age": customer_age,
                "support_ticket_count": support_ticket_count,
                "loan_type": loan_type,
            }

            for k, v in val_map.items():
                if k in input_data.columns:
                    input_data[k] = v

            if "loan_type" in input_data.columns:
                input_data["loan_type"] = input_data["loan_type"].astype("category")

            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

            st.markdown("---")
            st.subheader("Hasil Evaluasi Kredit:")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if pred == 1 or prob >= 0.5:
                    st.error(f"🚨 **RISIKO TINGGI: POTENSI DEFAULT**\n\nProbabilitas Gagal Bayar: **{prob:.2%}**")
                else:
                    st.success(f"✅ **RISIKO RENDAH: LAYAK DISETUJUI**\n\nProbabilitas Gagal Bayar: **{prob:.2%}**")

            with res_col2:
                st.metric("Debt-to-Income (DTI) Ratio", f"{dti:.2f}")
                st.metric("Skor Probabilitas Default", f"{prob:.4f}")
