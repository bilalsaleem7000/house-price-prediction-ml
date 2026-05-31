# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart House AI", layout="wide")

# =========================
# MODERN DARK UI
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0e1117, #1c1f26);
    color: white;
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
}
h1, h2, h3 {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df = df.dropna(thresh=len(df)*0.5, axis=1)

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()
X = df.drop("SalePrice", axis=1)
y = np.log1p(df["SalePrice"])

# =========================
# MODEL
# =========================
@st.cache_resource
def train_model():
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model()

# =========================
# HEADER
# =========================
st.markdown("""
<div style="text-align:center; padding:30px;">
    <h1>🏡 Smart House AI</h1>
    <p style="color:gray;">AI-powered property valuation system</p>
</div>
""", unsafe_allow_html=True)

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1,1])

# =========================
# INPUT PANEL
# =========================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🏠 Property Details")

    overall_qual = st.slider("Quality", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    total_bsmt = st.number_input("Basement Area", 0, 3000, 800)

    st.subheader("🚗 Additional Features")

    # Garage (dropdown + custom)
    garage_option = st.selectbox("Garage Capacity", ["0","1","2","3","4","Custom"])
    if garage_option == "Custom":
        garage_cars = st.number_input("Enter Garage Capacity", 0, 10, 2)
    else:
        garage_cars = int(garage_option)

    # =========================
    # YEAR BUILT (UPDATED LOGIC)
    # =========================
    st.markdown("### 🏗️ Year Built")

    year_option = st.selectbox(
        "Select Year (Modern Homes)",
        [str(y) for y in range(1980, 2011)] + ["Older Home (Enter Manually)"]
    )

    if year_option == "Older Home (Enter Manually)":
        year_built = st.number_input(
            "Enter Year Built (1872–1979)",
            min_value=1872,
            max_value=1979,
            value=1950
        )
    else:
        year_built = int(year_option)

    # Guidance note
    st.caption("ℹ️ Dataset range: 1872–2010. For houses built before 1980, select 'Older Home' and enter manually.")

    # Extra warning
    if year_built < 1900:
        st.warning("⚠️ Very old property. Prediction may be less accurate.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREP INPUT
# =========================
input_data = pd.DataFrame({
    'OverallQual': [overall_qual],
    'GrLivArea': [gr_liv_area],
    'GarageCars': [garage_cars],
    'TotalBsmtSF': [total_bsmt],
    'YearBuilt': [year_built]
})

input_data = input_data.reindex(columns=X.columns, fill_value=0)

# =========================
# PREDICTION
# =========================
prediction = model.predict(input_data)
prediction = np.expm1(prediction)[0]

# =========================
# OUTPUT PANEL
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("💰 Estimated Price")

    st.metric(label="Prediction", value=f"${prediction:,.0f}")

    # 📈 GAUGE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        gauge={
            'axis': {'range': [0, 500000]},
            'bar': {'color': "#00ffcc"},
            'steps': [
                {'range': [0,150000], 'color': "#1f3b4d"},
                {'range': [150000,300000], 'color': "#2a9d8f"},
                {'range': [300000,500000], 'color': "#52b788"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Category
    if prediction < 150000:
        st.info("💵 Budget Property")
    elif prediction < 300000:
        st.info("🏡 Mid-range Property")
    else:
        st.success("🏰 Luxury Property")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; padding:20px; color:gray;">
Built like a startup product 🚀
</div>
""", unsafe_allow_html=True)