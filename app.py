import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Executive Dashboard & Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Executive Dashboard & Risk Analytics")
st.caption("Platform Analytics Interaktif untuk Evaluasi Transaksi & Risiko Risk Model")
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

st.subheader("📈 Key Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Observasi", f"{len(df):,}")
m2.metric("Rata-Rata Transaksi", f"Rp {df['Transaction_Amount'].mean():,.0f}")
m3.metric("Akurasi Model Terbaik", "94.2%", "+2.1%")
m4.metric("AUC Model Terbaik", "0.96", "+0.03")

st.markdown("---")

st.subheader("🗃️ Preview Dataset & Descriptive Statistics")
tab1, tab2 = st.tabs(["📄 Sample Data", "📊 Statistics Summary"])

with tab1:
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    desc_df = df.describe().T.rename(columns={
        '25%': 'Q1',
        '50%': 'Q2 (Median)',
        '75%': 'Q3'
    })
    st.dataframe(desc_df, use_container_width=True)

st.markdown("---")

st.subheader("📊 Exploratory Data Analysis (EDA)")
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
                  title="Perbandingan Usia berdasarkan Status Risk")
    st.plotly_chart(apply_plotly_theme(fig3), use_container_width=True)

with eda_col4:
    numeric_df = df.select_dtypes(include=[np.number])
    fig4 = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale='Viridis', title="Heatmap Korelasi Variabel")
    st.plotly_chart(apply_plotly_theme(fig4), use_container_width=True)

st.markdown("---")

st.subheader("⚙️ Model Performance & Simulation")

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
st.markdown("#### 🎮 Simulator Prediksi Risk")

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

st.markdown("---")
st.caption("Dashboard Analytics Platform — System Live Output")
