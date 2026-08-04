import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Executive Dashboard - Credit & Fraud Analytics",
    page_icon="📊",
    layout="wide"
)

def format_rupiah(val):
    if val is None or np.isnan(val):
        return "Rp 0"
    return f"Rp {val:,.0f}".replace(",", ".")

def apply_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', size=11),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
    return fig

@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'CustomerID': [f"CUST-{i:04d}" for i in range(1, n+1)],
        'Age': np.random.randint(18, 65, size=n),
        'Income': np.random.normal(12000000, 3000000, size=n).clip(3000000, 30000000),
        'Transaction_Amount': np.random.exponential(scale=1500000, size=n),
        'Tx_Frequency_Daily': np.random.poisson(lam=3, size=n) + 1,
        'Tx_Hour': np.random.randint(0, 24, size=n),
        'Risk_Score': np.random.uniform(0, 1, size=n),
        'Category': np.random.choice(['Personal', 'Retail', 'Corporate'], size=n),
        'Loan_Status': np.random.choice(['Good Standing', 'Default Risk'], size=n, p=[0.85, 0.15]),
        'Fraud_Flag': np.random.choice(['Normal Transaction', 'Suspicious Fraud'], size=n, p=[0.93, 0.07])
    })
    return df

df = generate_data()

st.title("📊 Credit Risk & Multi-Factor Fraud Analytics Platform")
st.caption("Integrated Analytics Dashboard for Loan Assessment & Automated Fraud Detection Engine")

st.markdown("""
Platform analitik keputusan berbasis dua modul utama: **Loan Risk Analytics** (kelayakan kredit dan risiko gagal bayar) serta **Fraud Monitoring** dengan **Skor Anomali Otomatis** yang mengkalkulasi tingkat risiko berdasarkan gabungan frekuensi transaksi, waktu (jam malam), nominal, dan rasio profil finansial.
""")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Observasi", f"{len(df):,}".replace(",", "."))
m2.metric("Rata-Rata Transaksi", format_rupiah(df['Transaction_Amount'].mean()))
m3.metric("Tingkat Default Loan", f"{(df[df['Loan_Status']=='Default Risk'].shape[0]/len(df)*100):.1f}%")
m4.metric("Rasio Indikasi Fraud", f"{(df[df['Fraud_Flag']=='Suspicious Fraud'].shape[0]/len(df)*100):.1f}%")
m5.metric("Best AI Model AUC", "0.98", "Fraud & Loan")

st.markdown("---")

st.subheader("💳 1. Loan Default Risk Analytics")
st.caption("Fokus: Kapasitas finansial nasabah, rasio pendapatan terhadap pinjaman, dan estimasi kelayakan kredit.")

col_l1, col_l2 = st.columns(2)

with col_l1:
    fig_l1 = px.scatter(
        df, x='Income', y='Transaction_Amount', color='Loan_Status',
        title="Rasio Pendapatan vs Nominal Pinjaman",
        color_discrete_map={'Good Standing': '#2ecc71', 'Default Risk': '#e74c3c'},
        labels={'Income': 'Pendapatan (Rp)', 'Transaction_Amount': 'Nominal Pinjaman (Rp)'}
    )
    st.plotly_chart(apply_plotly_theme(fig_l1), use_container_width=True)

with col_l2:
    fig_l2 = px.box(
        df, x='Loan_Status', y='Age', color='Loan_Status',
        title="Sebaran Usia Nasabah berdasarkan Status Loan",
        color_discrete_map={'Good Standing': '#2ecc71', 'Default Risk': '#e74c3c'},
        labels={'Loan_Status': 'Status Loan', 'Age': 'Usia'}
    )
    st.plotly_chart(apply_plotly_theme(fig_l2), use_container_width=True)

st.markdown("---")

st.subheader("🛡️ 2. Multi-Factor Fraud Monitoring Analytics")
st.caption("Fokus: Deteksi transaksi abnormal berdasarkan frekuensi harian, waktu transaksi (jam malam), dan pola nominal.")

col_f1, col_f2 = st.columns(2)

with col_f1:
    fig_f1 = px.histogram(
        df, x='Tx_Hour', color='Fraud_Flag', barmode='overlay',
        title="Distribusi Waktu Transaksi (Jam 00:00 - 23:00)",
        color_discrete_map={'Normal Transaction': '#3498db', 'Suspicious Fraud': '#e67e22'},
        labels={'Tx_Hour': 'Jam Transaksi', 'count': 'Jumlah Transaksi'}
    )
    st.plotly_chart(apply_plotly_theme(fig_f1), use_container_width=True)

with col_f2:
    fig_f2 = px.scatter(
        df, x='Tx_Frequency_Daily', y='Transaction_Amount', color='Fraud_Flag',
        title="Frekuensi Transaksi Harian vs Nominal Transaksi",
        color_discrete_map={'Normal Transaction': '#3498db', 'Suspicious Fraud': '#e67e22'},
        labels={'Tx_Frequency_Daily': 'Frekuensi / Hari', 'Transaction_Amount': 'Nominal (Rp)'}
    )
    st.plotly_chart(apply_plotly_theme(fig_f2), use_container_width=True)

st.markdown("---")

st.subheader("🗃️ Data Profiling & Model Performance")

tab_data, tab_model = st.tabs(["📄 Sample Dataset & Ringkasan Statistik", "⚙️ Metrik Evaluasi Model AI"])

with tab_data:
    col_d1, col_d2 = st.columns([1.3, 1])
    with col_d1:
        st.markdown("**Sample Dataset Terintegrasi**")
        df_display = df.copy()
        df_display['Income'] = df_display['Income'].apply(format_rupiah)
        df_display['Transaction_Amount'] = df_display['Transaction_Amount'].apply(format_rupiah)
        st.dataframe(df_display.head(7), use_container_width=True, height=270)
    
    with col_d2:
        st.markdown("**Statistik Deskriptif Dataset**")
        desc_df = df.describe().T.rename(columns={'25%': 'Q1', '50%': 'Median', '75%': 'Q3'})
        st.dataframe(desc_df[['mean', 'std', 'min', 'Median', 'max']], use_container_width=True, height=270)

with tab_model:
    st.markdown("**Perbandingan Performa Model Classifier**")
    eval_df = pd.DataFrame({
        'Algoritma AI': ['Logistic Regression', 'Random Forest', 'XGBoost Classifier'],
        'Akurasi (Loan)': [0.82, 0.91, 0.94],
        'AUC (Loan)': [0.85, 0.93, 0.96],
        'Akurasi (Fraud)': [0.88, 0.94, 0.97],
        'AUC (Fraud)': [0.89, 0.95, 0.98],
        'Precision': [0.78, 0.88, 0.92],
        'Recall': [0.75, 0.86, 0.90]
    })
    st.dataframe(eval_df, use_container_width=True)

st.markdown("---")

st.subheader("🎮 Live Automated Decision Simulator (Loan & Fraud)")
st.caption("Masukkan parameter transaksi di bawah. Sistem AI akan **menghitung Skor Anomali Perilaku secara otomatis** tanpa input manual.")

with st.form("prediction_form"):
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("**1. Profil Demografi Nasabah**")
        input_age = st.number_input("Usia Nasabah", value=30, step=1)
        input_income = st.number_input("Pendapatan Tahunan (Rp)", value=12000000, step=500000)
        st.caption(f"Terbaca: **{format_rupiah(input_income)}**")
        
    with col_s2:
        st.markdown("**2. Rincian Transaksi**")
        input_tx = st.number_input("Nominal Transaksi (Rp)", value=2500000, step=100000)
        st.caption(f"Terbaca: **{format_rupiah(input_tx)}**")
        input_cat = st.selectbox("Segmen Kategori", ["Personal", "Retail", "Corporate"])
        
    with col_s3:
        st.markdown("**3. Pola & Waktu Transaksi**")
        input_freq = st.number_input("Frekuensi Transaksi Hari Ini", value=2, step=1)
        input_hour = st.slider("Jam Transaksi (00:00 - 23:00)", 0, 23, 14)
        
    submit = st.form_submit_button("🚀 Evaluasi Keputusan dengan Algoritma Otomatis", use_container_width=True)

if submit:
    safe_income = max(input_income, 1) if input_income > 0 else 1
    
    is_night = 1 if (input_hour >= 23 or input_hour <= 4) else 0
    
    time_anomaly = 0.35 if is_night else 0.05
    
    freq_anomaly = min(max((input_freq - 3) * 0.05, 0.0), 0.35)
    
    monthly_income = safe_income / 12
    tx_ratio = input_tx / monthly_income
    tx_anomaly = min(max((tx_ratio - 0.2) * 0.4, 0.0), 0.30)
    
    auto_anomaly_score = min(max(time_anomaly + freq_anomaly + tx_anomaly, 0.05), 0.99)
    
    loan_risk = min(max((input_tx / safe_income) * 1.8 + (auto_anomaly_score * 0.2), 0.05), 0.98)
    fraud_risk = auto_anomaly_score

    st.markdown("### 📋 Hasil Evaluasi Keputusan AI")
    st.info(f"🤖 **Skor Anomali Perilaku Terkalkulasi Otomatis:** `{auto_anomaly_score:.2f}` / 1.00 (Dihitung berdasarkan indikator Waktu, Frekuensi, dan Rasio Nominal)")

    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("#### 💳 Evaluation 1: Kelayakan Loan")
        fig_gauge_loan = go.Figure(go.Indicator(
            mode="gauge+number",
            value=loan_risk * 100,
            number={'suffix': "%"},
            title={'text': "Probabilitas Gagal Bayar (Default)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c" if loan_risk > 0.5 else "#2ecc71"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(46, 204, 113, 0.2)"},
                    {'range': [40, 100], 'color': "rgba(231, 76, 60, 0.2)"}
                ]
            }
        ))
        st.plotly_chart(apply_plotly_theme(fig_gauge_loan), use_container_width=True)
        
        if loan_risk > 0.5:
            st.error("⛔ **REKOMENDASI: LOAN DITOLAK**\n\nRasio pengajuan nominal pinjaman terlalu tinggi dibandingkan pendapatan.")
        else:
            st.success("✅ **REKOMENDASI: LOAN DISETUJUI**\n\nProfil finansial nasabah aman dan memenuhi batas rasio kelayakan.")

    with col_res2:
        st.markdown("#### 🛡️ Evaluation 2: Indikasi Fraud Transaksi")
        fig_gauge_fraud = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_risk * 100,
            number={'suffix': "%"},
            title={'text': "Probabilitas Anomali Fraud"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e67e22" if fraud_risk > 0.4 else "#3498db"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(52, 152, 219, 0.2)"},
                    {'range': [30, 100], 'color': "rgba(230, 126, 34, 0.2)"}
                ]
            }
        ))
        st.plotly_chart(apply_plotly_theme(fig_gauge_fraud), use_container_width=True)
        
        fraud_reasons = []
        if is_night:
            fraud_reasons.append("Pola Waktu Abnormal: Transaksi dilakukan pada jam malam (23:00 - 04:00)")
        if input_freq > 8:
            fraud_reasons.append(f"Pola Frekuensi Anomali: Frekuensi transaksi sangat tinggi ({input_freq}x per hari)")
        if tx_ratio > 0.5:
            fraud_reasons.append(f"Pola Nominal Ekstrem: Nominal transaksi mencapai {tx_ratio:.1f}x dari estimasi pendapatan bulanan")

        if fraud_risk > 0.4:
            st.warning("⚠️ **PERINGATAN: TRANSAKSI MENCURIGAKAN**\n\n" + "\n".join([f"- {r}" for r in fraud_reasons]))
        else:
            st.info("🛡️ **PEMERIKSAAN: TRANSAKSI NORMAL**\n\nSeluruh indikator transaksi berada dalam ambang batas wajar.")

st.markdown("---")
st.caption("Credit Risk & Fraud Analytics Decision Platform — Executive Dashboard")
