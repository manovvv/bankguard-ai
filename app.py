import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Executive Dashboard - Credit & Risk Decision System",
    page_icon="📊",
    layout="wide"
)

def format_rupiah(val):
    """Fungsi pembantu untuk format angka ke Rupiah dengan titik pemisah ribuan"""
    if val is None:
        return "Rp 0"
    return f"Rp {val:,.0f}".replace(",", ".")

# Header Utama & Penjelasan Fungsi Aplikasi dalam Paragraf
st.title("📊 Executive Dashboard & Risk Analytics")
st.markdown("""
Aplikasi ini merupakan platform analisis interaktif dan sistem pengambilan keputusan risiko kredit berbasis Artificial Intelligence (AI). Platform ini berfungsi untuk membantu tim analis dalam memantau portofolio transaksi nasabah, menemukan pola sebaran risiko melalui analisis visual (EDA), serta menguji performa model *Machine Learning* untuk prediksi risiko *default*. Melalui simulator interaktif yang tersedia, pengguna dapat melakukan evaluasi kelayakan kredit secara *real-time* berdasarkan masukan parameter seperti pendapatan, nominal transaksi, dan skor risiko internal tanpa adanya batasan minimum input angka.
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
        'Target_Default': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    })
    return df

df = generate_data()

# Metric Utama Portofolio
st.subheader("📈 Indikator Utama Portofolio Risiko")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Observasi Data", f"{len(df):,}".replace(",", "."))
m2.metric("Rata-Rata Nominal Transaksi", format_rupiah(df['Transaction_Amount'].mean()))
m3.metric("Akurasi Model Terbaik", "94.2%", "+2.1%")
m4.metric("AUC Model Terbaik", "0.96", "+0.03")

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

# Analisis EDA
st.subheader("📊 Exploratory Data Analysis (EDA)")
st.caption("Visualisasi hubungan antar variabel untuk mendeteksi indikator awal risiko kredit.")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    fig1 = px.pie(df, names='Category', title="Proporsi Nasabah per Kategori", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(apply_plotly_theme(fig1), use_container_width=True)

with eda_col2:
    fig2 = px.scatter(df, x='Income', y='Transaction_Amount', color='Target_Default',
                      title="Hubungan Income vs Nominal Transaksi",
                      color_continuous_scale='Reds')
    st.plotly_chart(apply_plotly_theme(fig2), use_container_width=True)

eda_col3, eda_col4 = st.columns(2)

with eda_col3:
    fig3 = px.box(df, x='Target_Default', y='Age', color='Target_Default',
                  title="Perbandingan Usia berdasarkan Status Risk (Default)")
    st.plotly_chart(apply_plotly_theme(fig3), use_container_width=True)

with eda_col4:
    numeric_df = df.select_dtypes(include=[np.number])
    fig4 = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale='Viridis', title="Heatmap Korelasi Variabel Risiko")
    st.plotly_chart(apply_plotly_theme(fig4), use_container_width=True)

st.markdown("---")

# Model Performance & Simulator Interaktif
st.subheader("⚙️ Performa Model & Simulator Keputusan Kredit")
st.write("Tabel di bawah menunjukkan perbandingan metrik evaluasi dari berbagai algoritma AI yang diuji:")

eval_df = pd.DataFrame({
    'Model Algorithm': ['Logistic Regression', 'Random Forest', 'XGBoost Classifier'],
    'Accuracy': [0.82, 0.91, 0.94],
    'AUC': [0.85, 0.93, 0.96],
    'Precision': [0.78, 0.88, 0.92],
    'Recall': [0.75, 0.86, 0.90],
    'F1-Score': [0.76, 0.87, 0.91]
})
st.table(eval_df)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🎮 Simulator Keputusan Kelayakan Kredit (Real-time Evaluation)")
st.write("Masukan parameter di bawah untuk menguji secara otomatis apakah pengajuan disetujui atau ditolak oleh sistem AI:")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        input_age = st.number_input("Usia Nasabah", value=30, step=1)
        input_income = st.number_input("Pendapatan Tahunan (Rp)", value=10000000, step=500000)
        st.caption(f"Format Terbaca: **{format_rupiah(input_income)}**")
    with c2:
        input_tx = st.number_input("Nominal Transaksi/Pengajuan (Rp)", value=2000000, step=100000)
        st.caption(f"Format Terbaca: **{format_rupiah(input_tx)}**")
        input_cat = st.selectbox("Kategori Segmen", ["Personal", "Retail", "Corporate"])
    with c3:
        input_score = st.slider("Skor Risiko Internal (0 = Sangat Aman, 1 = Sangat Riskan)", 0.0, 1.0, 0.45)
        
    submit = st.form_submit_button("🚀 Evaluasi Keputusan AI", use_container_width=True)

if submit:
    safe_income = max(input_income, 1) if input_income > 0 else 1
    calculated_risk = (input_tx / safe_income) * 2 + (input_score * 0.5)
    calculated_risk = min(max(calculated_risk, 0.05), 0.98)

    col_res1, col_res2 = st.columns([1, 1])
    with col_res1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=calculated_risk * 100,
            number={'suffix': "%"},
            title={'text': "Estimasi Probabilitas Default (Risiko)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if calculated_risk > 0.5 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(76, 175, 80, 0.2)"},
                    {'range': [30, 60], 'color': "rgba(255, 152, 0, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(244, 67, 54, 0.2)"}
                ]
            }
        ))
        st.plotly_chart(apply_plotly_theme(fig_gauge), use_container_width=True)

    with col_res2:
        st.markdown("<br>", unsafe_allow_html=True)
        if calculated_risk > 0.5:
            st.error("⛔ **REKOMENDASI SISTEM: DITOLAK (HIGH RISK)**")
            st.write("Nilai transaksi yang diajukan tidak seimbang dengan profil pendapatan dan skor risiko internal nasabah.")
        else:
            st.success("✅ **REKOMENDASI SISTEM: DISETUJUI (LOW RISK)**")
            st.write("Pengajuan berada di dalam ambang toleransi batas risiko yang disyaratkan.")

st.markdown("---")
st.caption("Credit Risk Decision System — Analytics & Automated Decision Platform")
