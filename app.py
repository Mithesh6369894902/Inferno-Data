import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="InfernoData", layout="wide")

st.title("InfernoData")
st.caption("Dataset Engineering & ML Validation")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Dataset Generator",
        "Dataset Trimmer",
        "Classification",
        "Regression",
        "Clustering",
        "Association Rule Mining"
    ]
)

def download_csv(df, name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="{name}">Download CSV</a>', unsafe_allow_html=True)

# ---------------- HOME ----------------
if page == "Home":
    st.write("InfernoData focuses on dataset preparation and ML validation.")

# ---------------- DATASET GENERATOR ----------------
elif page == "Dataset Generator":
    rows = st.slider("Rows", 10, 500, 100)
    cols = st.slider("Columns", 2, 10, 4)

    if st.button("Generate"):
        data = np.random.randn(rows, cols)
        df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(cols)])
        st.dataframe(df.head())
        download_csv(df, "synthetic.csv")

# ---------------- DATASET TRIMMER ----------------
elif page == "Dataset Trimmer":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        cols = st.multiselect("Columns", df.columns)
        rows = st.slider("Rows", 1, len(df), min(100, len(df)))

        if st.button("Trim"):
            trimmed = df[cols].sample(rows, replace=True)
            st.dataframe(trimmed.head())
            download_csv(trimmed, "trimmed.csv")

# ---------------- CLASSIFICATION ----------------
elif page == "Classification":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        target = st.selectbox("Target", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.text(classification_report(y_test, preds))

# ---------------- REGRESSION ----------------
elif page == "Regression":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        target = st.selectbox("Target", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.write("MAE:", mean_absolute_error(y_test, preds))
        st.write("MSE:", mean_squared_error(y_test, preds))
        st.write("R2:", r2_score(y_test, preds))

# ---------------- CLUSTERING ----------------
elif page == "Clustering":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        k = st.slider("Clusters", 2, 10, 3)

        model = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = model.fit_predict(df)

        st.dataframe(df.head())

# ---------------- ASSOCIATION RULE MINING ----------------
elif page == "Association Rule Mining":
    file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        support = st.slider("Min Support", 0.01, 0.5, 0.05)
        confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5)

        if st.button("Generate Rules"):
            transactions = df.iloc[:, 0].dropna().astype(str).apply(lambda x: x.split(",")).tolist()

            te = TransactionEncoder()
            te_array = te.fit(transactions).transform(transactions)
            df_bin = pd.DataFrame(te_array, columns=te.columns_)

            freq = apriori(df_bin, min_support=support, use_colnames=True)

            if freq.empty:
                st.warning("No frequent itemsets.")
            else:
                rules = association_rules(freq, metric="confidence", min_threshold=confidence)
                st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
                download_csv(rules, "association_rules.csv")


