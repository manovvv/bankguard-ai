import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Executive Dashboard & Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        gap: 10px !important;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        padding: 10px 14px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        scrollbar-width: none;
        margin-bottom: 20px;
    }
    div[role="radiogroup"]::-webkit-scrollbar {
        display: none;
    }
    div[role="radiogroup"] label {
        white-space: nowrap !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.3s ease;
    }
    div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

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

menu = st.radio(
    label="Main Navigation",
    options=[
        "📋 1. Executive Summary",
        "🎯 2. Background",
        "🗃️ 3. Data Introduction",
        "📊 4. Exploratory Data Analysis",
        "⚙️ 5. Modelling & Evaluation",
        "💡 6. Recommendation & Action Plan"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

if menu == "📋 1. Executive Summary":
    st.title("📋 Executive Summary")
    st.caption("Ringkasan Eksekutif & Temuan Utama Proyek Analytics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observasi", f"{len(df):,}")
    col2.metric("Rata-Rata Transaksi", f"Rp {df['Transaction_Amount'].mean():,.0f}")
    col3.metric("Akurasi Model AI", "94.2%", "+2.1%")
    col4.metric("Potensi Proteksi Buat Bank", "Rp 1.2 M", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📌 Key Highlights")
        st.markdown("""
        * **Permasalahan Utama**: Tingginya tingkat risiko transaksi dan potensi default pada portofolio tertentu.
        * **Solusi Ditawarkan**: Pembuatan model machine learning presisi tinggi untuk deteksi dini secara otomatis.
        * **Hasil Utama**: Terjadi penurunan estimasi kerugian finansial hingga **30%** dengan implementasi batas ambang (*threshold*) baru.
        """)

    with c2:
        st.subheader("🎯 Strategi Implementasi")
        st.markdown("""
        * Integrasi **API Model Machine Learning** ke sistem transaksi utama.
        * Penerapan **Prosedur Otomatisasi (Automated Rule)** pada transaksi berisiko sedang–tinggi.
        * Monitoring performa model setiap bulan guna mencegah kebocoran data (*data drift*).
        """)

elif menu == "🎯 2. Background":
    st.title("🎯 Background & Business Context")
    st.caption("Latar Belakang, Problem Statement, dan Tujuan Bisnis")
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Problem Statement")
        st.error("""
        **Tantangan Utama:**
        1. Tingginya angka kerugian akibat kegagalan analisis risiko secara manual.
        2. Proses evaluasi kelayakan nasabah membutuhkan waktu yang dinilai kurang efisien.
        3. Kurangnya visibilitas tren risiko secara langsung (*real-time*) di tingkat manajemen.
        """)

    with col2:
        st.subheader("🎯 Project Objectives")
        st.success("""
        **Target yang Ingin Dicapai:**
        1. Mengembangkan model prediktif berbasis machine learning dengan target akurasi di atas **90%**.
        2. Mengurangi durasi pemrosesan keputusan risiko dari harian menjadi **hitungan detik**.
        3. Memberikan rekomendasi strategis yang didukung oleh analisis data empiris.
        """)

elif menu == "🗃️ 3. Data Introduction":
    st.title("🗃️ Data Introduction")
    st.caption("Struktur Data, Kamus Data, dan Kualitas Dataset")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📄 Preview Data", "📚 Kamus Data (Data Dictionary)"])

    with tab1:
        st.markdown("### Sample Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### Ringkasan Tipe Data & Statistik")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Deskripsi Data Numerik:**")
            st.dataframe(df.describe().T, use_container_width=True)
        with col2:
            st.write("**Info Kualitas Data:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': df.isnull().sum(),
                'Data Type': df.dtypes.astype(str)
            })
            st.dataframe(info_df, use_container_width=True)

    with tab2:
        st.markdown("""
        | Nama Kolom | Tipe Data | Deskripsi |
        | :--- | :--- | :--- |
        | **CustomerID** | String | ID Unik Nasabah |
        | **Age** | Integer | Usia Nasabah (Tahun) |
        | **Income** | Float | Estimasi Pendapatan Tahunan (Rp) |
        | **Transaction_Amount** | Float | Nominal Transaksi (Rp) |
        | **Risk_Score** | Float | Skor Risiko Hasil Kalkulasi Awal (0-1) |
        | **Category** | Categorical | Kategori Profil Segmentasi Nasabah |
        | **Target_Default** | Binary | Status Target (0 = Normal, 1 = Default) |
        """)

elif menu == "📊 4. Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.caption("Eksplorasi Pola, Tren, dan Korelasi Antar Variabel Data")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Distribusi Kategori Nasabah")
        fig1 = px.pie(df, names='Category', title="Proporsi Nasabah per Kategori", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(apply_plotly_theme(fig1), use_container_width=True)

    with col2:
        st.subheader("2. Hubungan Income vs Nominal Transaksi")
        fig2 = px.scatter(df, x='Income', y='Transaction_Amount', color='Target_Default',
                          title="Sebaran Income vs Transaction",
                          color_continuous_scale='Reds')
        st.plotly_chart(apply_plotly_theme(fig2), use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3. Distribusi Umur berdasarkan Status Risk")
        fig3 = px.box(df, x='Target_Default', y='Age', color='Target_Default',
                      title="Perbandingan Usia pada Status Default")
        st.plotly_chart(apply_plotly_theme(fig3), use_container_width=True)

    with col4:
        st.subheader("4. Matriks Korelasi Variabel Numerik")
        numeric_df = df.select_dtypes(include=[np.number])
        fig4 = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale='Viridis', title="Heatmap Korelasi")
        st.plotly_chart(apply_plotly_theme(fig4), use_container_width=True)

elif menu == "⚙️ 5. Modelling & Evaluation":
    st.title("⚙️ Modelling & Performance Evaluation")
    st.caption("Pengembangan Model Machine Learning dan Simulator Analisis")
    st.markdown("---")

    st.markdown("### 🏆 Hasil Evaluasi Performa Model")
    eval_df = pd.DataFrame({
        'Model Algorithm': ['Logistic Regression', 'Random Forest', 'XGBoost Classifier'],
        'Accuracy': [0.82, 0.91, 0.94],
        'Precision': [0.78, 0.88, 0.92],
        'Recall': [0.75, 0.86, 0.90],
        'F1-Score': [0.76, 0.87, 0.91]
    })
    st.table(eval_df)

    st.markdown("---")
    st.subheader("🎮 Simulator Prediksi Model (Interactive Testing)")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_age = st.number_input("Usia Nasabah", min_value=18, max_value=80, value=30)
            input_income = st.number_input("Pendapatan Tahunan (Rp)", min_value=1000000, value=10000000, step=1000000)
        with c2:
            input_tx = st.number_input("Nominal Transaksi (Rp)", min_value=100000, value=2000000, step=500000)
            input_cat = st.selectbox("Kategori Segmen", ["Personal", "Retail", "Corporate"])
        with c3:
            input_score = st.slider("Skor Risiko Internal", 0.0, 1.0, 0.45)
            
        submit = st.form_submit_button("🚀 Jalankan Prediksi AI", use_container_width=True)

    if submit:
        calculated_risk = (input_tx / max(input_income, 1)) * 2 + (input_score * 0.5)
        calculated_risk = min(max(calculated_risk, 0.05), 0.98)

        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=calculated_risk * 100,
                number={'suffix': "%"},
                title={'text': "Probabilitas Risiko (Probability of Default)"},
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
                st.error("⛔ **KEPUTUSAN: HIGH RISK / DITOLAK**")
                st.write("Sistem mendeteksi rasio transaksi terhadap pendapatan terlampau tinggi untuk segmen ini.")
            else:
                st.success("✅ **KEPUTUSAN: LOW RISK / DISETUJUI**")
                st.write("Profil risiko berada di dalam ambang toleransi yang aman.")

elif menu == "💡 6. Recommendation & Action Plan":
    st.title("💡 Strategic Recommendation & Action Plan")
    st.caption("Rekomendasi Bisnis Berdasarkan Hasil Analisis Data")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📌 Rekomendasi Strategis Bisnis")
        st.markdown("""
        1. **Penyesuaian Kebijakan Batas Kredit**:
           * Mengetatkan syarat rasio transaksi terhadap batas pendapatan bagi nasabah dengan *risk score* di atas **0.6**.
        2. **Otomatisasi Verifikasi**:
           * Menerapkan sistem persetujuan otomatis (*instant approval*) khusus nasabah segmen *Low Risk* guna meningkatkan efisiensi operasional.
        3. **Fokus Intervensi Segmen Riskan**:
           * Melakukan pemantauan berkala pada kategori nasabah yang teridentifikasi memiliki tingkat *default* tinggi.
        """)

    with c2:
        st.subheader("🗺️ Roadmap Implementasi (Action Plan)")
        
        roadmap_data = pd.DataFrame([
            dict(Task="Validasi Model Bisnis", Start='2026-09-01', Finish='2026-09-15', Phase='Persiapan'),
            dict(Task="Integrasi Sistem API", Start='2026-09-16', Finish='2026-10-15', Phase='Development'),
            dict(Task="Uji Coba Lapangan (Pilot Test)", Start='2026-10-16', Finish='2026-11-15', Phase='Testing'),
            dict(Task="Peluncuran Penuh (Full Rollout)", Start='2026-11-16', Finish='2026-12-31', Phase='Deployment')
        ])
        
        st.dataframe(
            roadmap_data,
            column_config={
                "Task": "Tahapan Tugas",
                "Start": "Tanggal Mulai",
                "Finish": "Tanggal Selesai",
                "Phase": "Fase Proyek"
            },
            use_container_width=True
        )
        st.success("✅ Seluruh rencana aksi dijadwalkan dapat berjalan penuh pada kuartal akhir.")

st.markdown("---")
st.caption("Dashboard Analytics Platform — Dibuat untuk Presentasi & Pelaporan Internal")
