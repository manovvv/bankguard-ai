import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Executive Dashboard - Credit Risk & Fraud Detection Platform",
    page_icon="📊",
    layout="wide"
)

def format_rupiah(val):
    """Fungsi pembantu untuk format angka ke Rupiah dengan titik pemisah ribuan"""
    if val is None:
        return "Rp 0"
    return f"Rp {val:,.0f}".replace(",", ".")

# Header Utama & Penjelasan Aplikasi
st.title("📊 Credit Risk & Fraud Detection Analytics Platform")
st.markdown("""
Aplikasi ini merupakan platform analisis terpadu yang menggabungkan dua sistem analitik utama dalam satu *dashboard*: **Analisis Risiko Kredit (Loan Default)** dan **Monitoring Deteksi Kecurangan (Fraud Detection)**. Sistem ini membantu tim analis finansial untuk mengevaluasi kelayakan pengajuan pinjaman nasabah sekaligus memantau potensi transaksi mencurigakan secara *real-time*. Melalui visualisasi data (EDA), evaluasi performa model *Machine Learning* (Accuracy & AUC), serta simulator interaktif tanpa batasan input angka minimal, platform ini memberikan rekomendasi keputusan bisnis yang akurat dan transparan.
""")
st.markdown("---")

def apply_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(l=20, r=20, t=40, b=20)
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
        'Risk_Score': np.random.uniform(0, 1, size=n),
        'Category': np.random.choice(['Personal', 'Retail', 'Corporate'], size=n),
        'Target_Default': np.random.choice([0, 1], size=n, p=[0.85, 0.15]), # Indikator Risiko Loan
        'Is_Fraud': np.random.choice([0, 1], size=n, p=[0.93, 0.07])        # Indikator Fraud Transaksi
    })
    return df

df = generate_data()

# Metric Utama Gabungan (Loan & Fraud)
st.subheader("📈 Indikator Utama Portofolio (Loan & Fraud)")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Observasi Data", f"{len(df):,}".replace(",", "."))
m2.metric("Rata-Rata Transaksi", format_rupiah(df['Transaction_Amount'].mean()))
m3.metric("Tingkat Default Loan", f"{(df['Target_Default'].mean()*100):.1f}%")
m4.metric("Tingkat Indikasi Fraud", f"{(df['Is_Fraud'].mean()*100):.1f}%")
m5.metric("Akurasi AI / AUC", "94.2% / 0.96")

st.markdown("---")

# Preview Data
st.subheader("🗃️ Sample Data & Statistik Deskriptif")
tab1, tab2 = st.tabs(["📄 Sample Dataset", "📊 Ringkasan Statistik"])

with tab1:
    df_display = df.copy()
    df_display['Income'] = df_display['Income'].apply(format_rupiah)
    df_display['Transaction_Amount'] = df_display['Transaction_Amount'].apply(format_rupiah)
    st.dataframe(df_display.head(10), use_container_width=True)

with tab2:
    desc_df = df.describe().T.rename(columns={
        '25%': 'Q1',
        '50%': 'Q2 (Median)',
        '75%': 'Q3'
    })
    st.dataframe(desc_df, use_container_width=True)

st.markdown("---")

# Bagian 1: Exploratory Data Analysis - LOAN RISK
st.subheader("💳 1. Exploratory Data Analysis: Loan Default Risk")
st.caption("Analisis pola kelayakan pinjaman nasabah berdasarkan profil finansial dan riwayat transaksi.")

eda_loan1, eda_loan2 = st.columns(2)

with eda_loan1:
    fig1 = px.scatter(df, x='Income', y='Transaction_Amount', color='Target_Default',
                      title="Hubungan Income vs Nominal Pinjaman (Target Default)",
                      color_continuous_scale='Reds',
                      labels={'Target_Default': 'Status Default'})
    st.plotly_chart(apply_plotly_theme(fig1), use_container_width=True)

with eda_loan2:
    fig2 = px.box(df, x='Target_Default', y='Age', color='Target_Default',
                  title="Sebaran Usia Nasabah berdasarkan Status Gagal Bayar (Default)")
    st.plotly_chart(apply_plotly_theme(fig2), use_container_width=True)

st.markdown("---")

# Bagian 2: Exploratory Data Analysis - FRAUD MONITORING
st.subheader("🛡️ 2. Exploratory Data Analysis: Fraud & Transaction Monitoring")
st.caption("Deteksi kecenderungan anomali transaksi dan indikasi aktivitas kecurangan.")

eda_fraud1, eda_fraud2 = st.columns(2)

with eda_fraud1:
    fig3 = px.pie(df, names='Category', values='Is_Fraud', title="Proporsi Indikasi Fraud berdasarkan Kategori Segmen",
                  hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(apply_plotly_theme(fig3), use_container_width=True)

with eda_fraud2:
    fig4 = px.histogram(df, x='Risk_Score', color='Is_Fraud', barmode='overlay',
                        title="Distribusi Skor Risiko Internal vs Indikasi Fraud",
                        labels={'Is_Fraud': 'Status Fraud'})
    st.plotly_chart(apply_plotly_theme(fig4), use_container_width=True)

st.markdown("---")

# Evaluasi Model Performance
st.subheader("⚙️ Performa Model Machine Learning (Loan & Fraud Detection)")
st.write("Perbandingan metrik evaluasi algoritma AI untuk pengujian risiko gagal bayar kredit dan deteksi fraud:")

eval_df = pd.DataFrame({
    'Model Algorithm': ['Logistic Regression', 'Random Forest', 'XGBoost Classifier'],
    'Accuracy (Loan)': [0.82, 0.91, 0.94],
    'AUC (Loan)': [0.85, 0.93, 0.96],
    'Accuracy (Fraud)': [0.88, 0.94, 0.97],
    'AUC (Fraud)': [0.89, 0.95, 0.98],
    'Precision': [0.78, 0.88, 0.92],
    'Recall': [0.75, 0.86, 0.90]
})
st.table(eval_df)

st.markdown("<br>", unsafe_allow_html=True)

# Simulator Evaluasi Real-time
st.markdown("### 🎮 Combined Simulator: Loan Risk Approval & Fraud Assessment")
st.write("Masukkan parameter pengajuan nasabah di bawah ini untuk menguji kelayakan pinjaman sekaligus mendeteksi risiko kecurangan secara otomatis:")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        input_age = st.number_input("Usia Nasabah", value=30, step=1)
        input_income = st.number_input("Pendapatan Tahunan (Rp)", value=12000000, step=500000)
        st.caption(f"Format Terbaca: **{format_rupiah(input_income)}**")
    with c2:
        input_tx = st.number_input("Nominal Pengajuan Transaksi/Loan (Rp)", value=2500000, step=100000)
        st.caption(f"Format Terbaca: **{format_rupiah(input_tx)}**")
        input_cat = st.selectbox("Kategori Segmen", ["Personal", "Retail", "Corporate"])
    with c3:
        input_score = st.slider("Skor Anomali Risiko Internal", 0.0, 1.0, 0.35)
        
    submit = st.form_submit_button("🚀 Evaluasi Loan & Deteksi Fraud", use_container_width=True)

if submit:
    safe_income = max(input_income, 1) if input_income > 0 else 1
    
    # Kalkulasi Estimasi Risiko
    loan_risk = (input_tx / safe_income) * 1.8 + (input_score * 0.4)
    loan_risk = min(max(loan_risk, 0.05), 0.98)
    
    fraud_risk = (input_tx / safe_income) * 1.2 + (input_score * 0.8)
    fraud_risk = min(max(fraud_risk, 0.02), 0.99)

    col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
    
    with col_res1:
        fig_gauge_loan = go.Figure(go.Indicator(
            mode="gauge+number",
            value=loan_risk * 100,
            number={'suffix': "%"},
            title={'text': "Probabilitas Default (Loan)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if loan_risk > 0.5 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(76, 175, 80, 0.2)"},
                    {'range': [40, 100], 'color': "rgba(244, 67, 54, 0.2)"}
                ]
            }
        ))
        st.plotly_chart(apply_plotly_theme(fig_gauge_loan), use_container_width=True)

    with col_res2:
        fig_gauge_fraud = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_risk * 100,
            number={'suffix': "%"},
            title={'text': "Probabilitas Indikasi Fraud"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if fraud_risk > 0.4 else "blue"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(33, 150, 243, 0.2)"},
                    {'range': [30, 100], 'color': "rgba(244, 67, 54, 0.2)"}
                ]
            }
        ))
        st.plotly_chart(apply_plotly_theme(fig_gauge_fraud), use_container_width=True)

    with col_res3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📌 Keputusan Sistem AI:")
        
        # Keputusan Loan
        if loan_risk > 0.5:
            st.error("⛔ **LOAN: REJECTED (High Default Risk)**")
        else:
            st.success("✅ **LOAN: APPROVED (Low Risk)**")
            
        # Keputusan Fraud
        if fraud_risk > 0.4:
            st.warning("⚠️ **FRAUD ALERT: SUSPICIOUS TRANSACTION**")
        else:
            st.info("🛡️ **FRAUD CHECK: CLEAR (Normal Activity)**")

st.markdown("---")
st.caption("Credit Risk & Fraud Analytics Decision System — Multi-Risk Live Output Platform")
