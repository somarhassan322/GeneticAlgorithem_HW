import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.compare_methods import run_all


st.set_page_config(page_title='GA Feature Selection - BIA601', layout='wide')
st.title("Feature Selection using Genetic Algorithm — BIA601 - C2")


uploaded = st.file_uploader("Upload CSV dataset (the file must contain a target column)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Data Preview:")
    st.write(df.head())
    target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
else:
    st.info("No file uploaded. Using default dataset.")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Number of features for comparison", min_value=1, max_value=max(1, X.shape[1]//2), value=max(1, X.shape[1]//4))

if st.button("Run Analysis"):
    with st.spinner("Running comparisons..."):
        results = run_all(X_scaled, y, k_features=k, verbose=False)
    st.success("Analysis completed")
    st.write("Summary Results:")
    rows = []
    for method, metrics in results.items():
        rows.append({'Method': method, 'Accuracy': metrics['accuracy'], 'F1 Score': metrics['f1'], 'Number of Features': metrics['n_features']})
    st.table(pd.DataFrame(rows))