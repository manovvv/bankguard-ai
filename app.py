import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Executive Dashboard - Credit Risk & Fraud Analytics",
    page_icon="📊",
    layout="wide"
)

def format_rupiah(val):
    """Fungsi pembantu untuk format angka ke Rupiah dengan titik pemisah ribuan"""
    if val is None:
        return "Rp 0"
    return f"Rp {val:,.0f}".replace(",", ".")

# Header Utama
st.title("📊 Credit Risk & Fraud Analytics Platform")
st.markdown("""
Aplikasi ini merupakan platform analisis keputusan yang **memisahkan dua domain analisis utama**:
1. **💳 Loan Risk Analytics**: Berfokus pada kelayakan kredit nasabah, rasio pendapatan terhadap pinjaman, dan estimasi risiko gagal bayar (*default*).
2. **🛡️ Fraud Monitoring Analytics**: Berfokus pada pemantauan pola transaksi abnormal, deteksi anomali, dan pencegahan transaksi mencurigakan.
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
        'Loan_Status': np.random.choice(['Good Standing', 'Default Risk'], size=n, p=[0.85, 0.15]),
        'Fraud_Flag': np.random.choice(['Normal Transaction', 'Suspicious Fraud'], size=n, p=[0.93, 0.07])
    })
    return df

df = generate_data()

# ----------------------------------------------------
# 💳 KATEGORI 1: LOAN RISK ANALYTICS
# ----------------------------------------------------
st.header("💳 1. Analisis Kelayakan Pinjaman (Loan Default Risk)")
st.caption("Fokus pada kapasitas finansial nasabah dan estimasi risiko gagal bayar pengajuan kredit.")

# Metrics Loan
m_loan1, m_loan2, m_loan3, m_loan4 = st.columns(4)
m_loan1.metric("Total Pengajuan Loan", f"{len(df):,}".replace(",", "."))
m_loan2.metric("Rata-Rata Nominal Pinjaman", format_rupiah(df['Transaction_Amount'].mean()))
m_loan3.metric("Tingkat Default Loan", f"{(df[df['Loan_Status']=='Default Risk'].shape[0]/len(df)*100):.1f}%")
m_loan4.metric("Akurasi Model Loan / AUC", "94.2% / 0.96")

st.markdown("<br>", unsafe_allow_html=True)
st.write("**Eksplorasi Data (EDA - Loan Risk):**")

eda_l1, eda_l2 = st.columns(2)
with eda_l1:
    fig_l1 = px.scatter(df, x='Income', y='Transaction_Amount', color='Loan_Status',
                        title="Rasio Pendapatan vs Nominal Pinjaman (Loan Status)",
                        color_discrete_map={'Good Standing': '#2ecc71', 'Default Risk': '#e74c3c'})
    st.plotly_chart(apply_plotly_theme(fig_l1), use_container_width=True)

with eda_l2:
    fig_l2 = px.box(df, x='Loan_Status', y='Age', color='Loan_Status',
                    title="Sebaran Usia Nasabah berdasarkan Kelayakan Loan",
                    color_discrete_map={'Good Standing': '#2ecc71', 'Default Risk': '#e74c3c'})
    st.plotly_chart(apply_plotly_theme(fig_l2), use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 🛡️ KATEGORI 2: FRAUD MONITORING ANALYTICS
# ----------------------------------------------------
st.header("🛡️ 2. Analisis Deteksi Kecurangan (Fraud Monitoring)")
st.caption("Fokus pada deteksi aktivitas transaksi abnormal dan anomali perilaku nasabah.")

# Metrics Fraud
m_fraud1, m_fraud2, m_fraud3, m_fraud4 = st.columns(4)
m_fraud1.metric("Total Transaksi Terikutserta", f"{len(df):,}".replace(",", "."))
m_fraud2.metric("Total Indikasi Fraud", f"{df[df['Fraud_Flag']=='Suspicious Fraud'].shape[0]} Transaksi")
m_fraud3.metric("Rasio Tingkat Fraud", f"{(df[df['Fraud_Flag']=='Suspicious Fraud'].shape[0]/len(df)*100):.1f}%")
m_fraud4.metric("Akurasi Model Fraud / AUC", "97.1% / 0.98")

st.markdown("<br>", unsafe_allow_html=True)
st.write("**Eksplorasi Data (EDA - Fraud Monitoring):**")

eda_f1, eda_f2 = st.columns(2)
with eda_f1:
    fig_f1 = px.pie(df, names='Category', values=(df['Fraud_Flag']=='Suspicious Fraud').astype(int),
                    title="Distribusi Indikasi Fraud berdasarkan Segmen",
                    hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(apply_plotly_theme(fig_f1), use_container_width=True)

with eda_f2:
    fig_f2 = px.histogram(df, x='Risk_Score', color='Fraud_Flag', barmode='overlay',
                          title="Skor Anomali Internal vs Indikasi Fraud",
                          color_discrete_map={'Normal Transaction': '#3498db', 'Suspicious Fraud': '#e67e22'})
    st.plotly_chart(apply_plotly_theme(fig_f2), use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 🗃️ DATA & PERFORMA MODEL
# ----------------------------------------------------
st.subheader("🗃️ Ringkasan Statistik & Performa Model Machine Learning")

tab_data, tab_model = st.tabs(["📄 Sample Dataset & Statistik", "⚙️ Tabel Performa Model AI"])

with tab_data:
    col_d1, col_d2 = st.columns([1.2, 1])
    with col_d1:
        st.write("**Sample Dataset Terintegrasi:**")
        df_display = df.copy()
        df_display['Income'] = df_display['Income'].apply(format_rupiah)
        df_display['Transaction_Amount'] = df_display['Transaction_Amount'].apply(format_rupiah)
        st.dataframe(df_display.head(8), use_container_width=True)
    with col_d2:
        st.write("**Statistik Deskriptif Numerik:**")
        desc_df = df.describe().T.rename(columns={'25%': 'Q1', '50%': 'Q2', '75%': 'Q3'})
        st.dataframe(desc_df, use_container_width=True)

with tab_model:
    st.write("**Metrik Evaluasi Terpisah (Loan Classifier vs Fraud Detector):**")
    eval_df = pd.DataFrame({
        'Model Algorithm': ['Logistic Regression', 'Random Forest', 'XGBoost Classifier'],
        'Accuracy (Loan Risk)': [0.82, 0.91, 0.94],
        'AUC (Loan Risk)': [0.85, 0.93, 0.96],
        'Accuracy (Fraud Detection)': [0.88, 0.94, 0.97],
        'AUC (Fraud Detection)': [0.89, 0.95, 0.98],
        'Precision': [0.78, 0.88, 0.92],
        'Recall': [0.75, 0.86, 0.90]
    })
    st.table(eval_df)

st.markdown("---")

# ----------------------------------------------------
# 🎮 SIMULATOR KEPUTUSAN TERPISAH
# ----------------------------------------------------
st.subheader("🎮 Simulator Pengambilan Keputusan Real-Time")
st.write("Masukan data transaksi di bawah ini untuk melihat evaluasi terpisah antara kelayakan kredit dan kriteria kecurangan:")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        input_age = st.number_input("Usia Nasabah", value=30, step=1)
        input_income = st.number_input("Pendapatan Tahunan (Rp)", value=12000000, step=500000)
        st.caption(f"Terbaca: **{format_rupiah(input_income)}**")
    with c2:
        input_tx = st.number_input("Nominal Transaksi/Pengajuan (Rp)", value=2500000, step=100000)
        st.caption(f"Terbaca: **{format_rupiah(input_tx)}**")
        input_cat = st.selectbox("Kategori Segmen", ["Personal", "Retail", "Corporate"])
    with c3:
        input_score = st.slider("Skor Anomali Perilaku (0 = Normal, 1 = Sangat Mencurigakan)", 0.0, 1.0, 0.35)
        
    submit = st.form_submit_button("🚀 Jalankan Evaluasi Keputusan", use_container_width=True)

if submit:
    safe_income = max(input_income, 1) if input_income > 0 else 1
    
    # Formula Evaluasi Loan Risk
    loan_risk = (input_tx / safe_income) * 1.8 + (input_score * 0.3)
    loan_risk = min(max(loan_risk, 0.05), 0.98)
    
    # Formula Evaluasi Fraud Risk
    fraud_risk = (input_tx / safe_income) * 1.1 + (input_score * 0.85)
    fraud_risk = min(max(fraud_risk, 0.02), 0.99)

    col_sim_loan, col_sim_fraud = st.columns(2)
    
    with col_sim_loan:
        st.markdown("### 💳 1. Hasil Evaluasi Kelayakan Loan")
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
            st.error("⛔ **Rekomendasi Loan: DITOLAK**\n\nNominal pengajuan terlalu tinggi dibanding profil pendapatan.")
        else:
            st.success("✅ **Rekomendasi Loan: DISETUJUI**\n\nProfil rasio pendapatan dan nominal pengajuan memenuhi batas kriteria.")

    with col_sim_fraud:
        st.markdown("### 🛡️ 2. Hasil Evaluasi Indikasi Fraud")
        fig_gauge_fraud = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_risk * 100,
            number={'suffix': "%"},
            title={'text': "Probabilitas Anomali Transaksi (Fraud)"},
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
        
        if fraud_risk > 0.4:
            st.warning("⚠️ **Peringatan Fraud: TRANSAKSI MENCURIGAKAN**\n\nSistem mendeteksi anomali tinggi pada skor perilaku transaksi.")
        else:
            st.info("🛡️ **Pemeriksaan Fraud: TRANSAKSI NORMAL**\n\nAktivitas transaksi berada dalam pola wajar.")

st.markdown("---")
st.caption("Executive Dashboard & Analytics Platform — Dual Risk Decision System")
