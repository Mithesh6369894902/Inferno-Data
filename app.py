import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

from mlxtend.frequent_patterns import apriori, association_rules

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="InfernoData",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 InfernoData")
st.caption("Advanced Dataset Engineering & ML Validation Platform")

# ---------------- SIDEBAR ---------------- #
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🧪 Dataset Generator",
        "✂️ Dataset Trimmer",
        "🧠 Classification Execution",
        "📉 Regression Execution",
        "🧩 Clustering Execution",
        "🔗 Association Rule Mining"
    ]
)

# ---------------- UTIL ---------------- #
def download_csv(df, name="dataset.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(
        f'<a href="data:file/csv;base64,{b64}" download="{name}">⬇️ Download CSV</a>',
        unsafe_allow_html=True
    )

# ---------------- HOME ---------------- #
if page == "🏠 Home":
    st.markdown("""
    ## 🔥 InfernoData
    
    **InfernoData** is a dataset-centric ML platform that bridges the gap between  
    **data preparation** and **model validation**.

    ### What makes it different?
    - Focus on **dataset engineering**
    - Lightweight ML execution for **validation**
    - Supports **Classification, Regression, Clustering & Association**
    - Designed for **research & academic projects**

    > *Data comes first. Models come second.*
    """)

# ---------------- DATASET GENERATOR ---------------- #
elif page == "🧪 Dataset Generator":
    st.header("🧪 Synthetic Dataset Generator")

    rows = st.slider("Rows", 10, 500, 100)
    cols = st.slider("Columns", 2, 10, 4)

    if st.button("🔥 Generate Dataset"):
        data = np.random.randn(rows, cols)
        df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(cols)])
        st.dataframe(df.head())
        download_csv(df, "synthetic_dataset.csv")

# ---------------- TRIMMER ---------------- #
elif page == "✂️ Dataset Trimmer":
    st.header("✂️ Dataset Trimmer")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Original Shape:", df.shape)

        cols = st.multiselect("Select Columns", df.columns)
        rows = st.slider("Rows", 1, len(df), min(100, len(df)))

        if st.button("Trim Dataset"):
            trimmed = df[cols].sample(rows, replace=True)
            st.dataframe(trimmed.head())
            download_csv(trimmed, "trimmed_dataset.csv")

# ---------------- CLASSIFICATION ---------------- #
elif page == "🧠 Classification Execution":
    st.header("🧠 Classification Validation")

    file = st.file_uploader("Upload Classification Dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        target = st.selectbox("Target Column", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model_type = st.radio("Model", ["Logistic Regression", "Decision Tree"])

        if st.button("Train & Validate"):
            model = LogisticRegression(max_iter=1000) if model_type == "Logistic Regression" else DecisionTreeClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2f}")
            st.text("Classification Report")
            st.text(classification_report(y_test, preds))

# ---------------- REGRESSION ---------------- #
elif page == "📉 Regression Execution":
    st.header("📉 Regression Validation")

    file = st.file_uploader("Upload Regression Dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        target = st.selectbox("Target Column", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model_type = st.radio("Model", ["Linear Regression", "Ridge Regression"])

        if st.button("Train & Validate"):
            model = LinearRegression() if model_type == "Linear Regression" else Ridge()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.metric("MAE", f"{mean_absolute_error(y_test, preds):.2f}")
            st.metric("MSE", f"{mean_squared_error(y_test, preds):.2f}")
            st.metric("R²", f"{r2_score(y_test, preds):.2f}")

# ---------------- CLUSTERING ---------------- #
elif page == "🧩 Clustering Execution":
    st.header("🧩 Clustering Validation")

    file = st.file_uploader("Upload Numeric Dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        k = st.slider("Clusters", 2, 10, 3)

        if st.button("Run KMeans"):
            model = KMeans(n_clusters=k, random_state=42)
            df["Cluster"] = model.fit_predict(df)

            st.dataframe(df.head())

            fig, ax = plt.subplots()
            ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df["Cluster"])
            ax.set_title("Cluster Visualization")
            st.pyplot(fig)

# ---------------- ASSOCIATION ---------------- #
elif page == "🔗 Association Rule Mining":
    st.header("🔗 Association Rule Mining")

    file = st.file_uploader("Upload Transaction Dataset (Binary)", type=["csv"])
    if file:
        df = pd.read_csv(file)

        support = st.slider("Min Support", 0.01, 0.5, 0.05)
        confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5)

        if st.button("Generate Rules"):
            freq = apriori(df, min_support=support, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=confidence)

            st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
