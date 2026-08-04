# 🛡️ BankGuard AI Platform

BankGuard AI adalah platform analitik risiko keuangan berbasis Machine Learning yang dirancang untuk deteksi dini fraud transaksi dan evaluasi risiko kredit pinjaman.

## 🚀 Fitur Utama

- **🚨 Fraud Detection Simulator**: Mengidentifikasi transaksi kartu mencurigakan secara real-time.
- **💳 Loan Default Risk Predictor**: Memprediksi probabilitas gagal bayar (*Credit Default*) pada pengajuan kredit nasabah.

## 📁 Berkas Proyek

- `app.py`: Aplikasi utama Streamlit.
- `requirements.txt`: Daftar dependensi Python.
- `fraud_model.pkl` & `fraud_features.pkl`: Model dan fitur deteksi fraud.
- `loan_model.pkl` & `loan_features.pkl`: Model dan fitur risiko pinjaman.

## 🛠️ Cara Menjalankan di Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
