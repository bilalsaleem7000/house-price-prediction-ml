import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("train.csv")

# Preprocessing (same as before)
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df, drop_first=True)

X = df.drop("SalePrice", axis=1)
y = np.log1p(df["SalePrice"])

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model.fit(X, y)

# UI
st.title("🏠 House Price Prediction")

# Example inputs (you can expand later)
overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.slider("Garage Cars", 0, 4, 1)

# Create input data
input_data = pd.DataFrame({
    'OverallQual': [overall_qual],
    'GrLivArea': [gr_liv_area],
    'GarageCars': [garage_cars]
})

# Align columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Prediction
prediction = model.predict(input_data)
prediction = np.expm1(prediction)

st.success(f"Estimated House Price: ${prediction[0]:,.2f}")