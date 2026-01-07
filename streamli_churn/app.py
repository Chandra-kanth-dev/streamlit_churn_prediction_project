import os
import urllib.error
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Telco Churn App",
    page_icon="📉",
    layout="wide"
)

# -----------------------------
# LOAD CSS
# -----------------------------
def load_css():
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

st.title("📉 Telco Customer Churn Application")
st.caption("Logistic Regression • Analysis + Prediction")

# -----------------------------
# LOAD & PREPROCESS DATA
# -----------------------------
@st.cache_data
def load_data():
    # ---- 1. Try local CSV (if exists) ----
    local_path = os.path.join(
        os.path.dirname(__file__),
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
        else:
            raise FileNotFoundError
    except (FileNotFoundError, OSError):
        # ---- 2. Fallback to online CSV ----
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        try:
            df = pd.read_csv(url)
        except urllib.error.HTTPError:
            st.error("❌ Unable to load dataset from both local and online sources.")
            st.stop()

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})

    # One-hot encode Contract & PaymentMethod
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df[['Contract', 'PaymentMethod']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(),
        index=df.index
    )

    df = pd.concat([df, encoded_df], axis=1)

    # Internet-related features
    for col in ['OnlineSecurity', 'TechSupport']:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})

    # Convert TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    return df


df = load_data()

# -----------------------------
# FEATURES & TARGET
# -----------------------------
features = [
    'tenure',
    'MonthlyCharges',
    'Contract_One year',
    'Contract_Two year',
    'PaperlessBilling',
    'PaymentMethod_Electronic check',
    'TechSupport',
    'OnlineSecurity'
]

X = df[features]
y = df['Churn']

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["📊 Model Analysis", "🔮 Churn Prediction"])

# =====================================================
# TAB 1: MODEL ANALYSIS
# =====================================================
with tab1:
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")
    with col2:
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(3.6, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("🔍 Feature Influence")

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient")

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    sns.barplot(
        x="Coefficient",
        y="Feature",
        data=coef_df,
        palette="Spectral",
        ax=ax2
    )
    ax2.axvline(0, color="black", linewidth=0.8)
    st.pyplot(fig2)

# =====================================================
# TAB 2: CHURN PREDICTION
# =====================================================
with tab2:
    st.subheader("🧾 Enter Customer Details")

    with st.form("churn_form"):
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges", 20.0, 120.0, 70.0)

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Other"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])

        submit = st.form_submit_button("Predict Churn")

    if submit:
        input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract_One year': int(contract == "One year"),
            'Contract_Two year': int(contract == "Two year"),
            'PaperlessBilling': int(paperless == "Yes"),
            'PaymentMethod_Electronic check': int(payment == "Electronic check"),
            'TechSupport': int(tech_support == "Yes"),
            'OnlineSecurity': int(online_security == "Yes")
        }

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("📌 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2f}")
        else:
            st.success(f"✅ Customer is likely to STAY\n\nProbability: {1 - probability:.2f}")

st.divider()
st.caption(f"Model Accuracy: {accuracy:.2f}")
