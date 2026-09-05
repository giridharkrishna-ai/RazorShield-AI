
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="RazorShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM UI
# ==================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-top: 5px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 15px;
}

.metric-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    text-align: center;
    min-height: 105px;
}

.metric-title {
    font-size: 14px;
    opacity: 0.7;
}

.metric-value {
    font-size: 26px;
    font-weight: 750;
    margin-top: 8px;
}

.result-card {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    text-align: center;
    min-height: 135px;
}

.result-title {
    font-size: 15px;
    opacity: 0.7;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.info-box {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 15px;
}

.footer {
    text-align: center;
    opacity: 0.6;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("razorshield_fraud_model.pkl")

# ==================================================
# RECREATE TEST DATA
# ==================================================

np.random.seed(42)
N = 10000

df = pd.DataFrame({
    "amount": np.random.lognormal(mean=7, sigma=1.0, size=N),
    "transaction_hour": np.random.randint(0, 24, N),
    "customer_age": np.random.randint(18, 70, N),
    "transactions_last_24h": np.random.poisson(3, N),
    "avg_transaction_amount": np.random.lognormal(mean=6.5, sigma=0.7, size=N),
    "distance_from_last_transaction": np.random.exponential(50, N),
    "is_new_device": np.random.randint(0, 2, N),
    "is_international": np.random.randint(0, 2, N),
    "failed_attempts": np.random.poisson(0.7, N),
    "merchant_risk": np.random.randint(1, 6, N)
})

risk_score_raw = (
    (df["amount"] > df["avg_transaction_amount"] * 4) * 2
    + (df["transaction_hour"] <= 4) * 1
    + (df["is_new_device"] == 1) * 2
    + (df["is_international"] == 1) * 1
    + (df["failed_attempts"] >= 3) * 2
    + (df["distance_from_last_transaction"] > 100) * 1
    + (df["transactions_last_24h"] > 8) * 1
    + (df["merchant_risk"] >= 4) * 2
)

fraud_probability = np.clip(risk_score_raw / 10, 0, 0.85)
df["is_fraud"] = np.random.binomial(1, fraud_probability)

features = [
    "amount",
    "transaction_hour",
    "customer_age",
    "transactions_last_24h",
    "avg_transaction_amount",
    "distance_from_last_transaction",
    "is_new_device",
    "is_international",
    "failed_attempts",
    "merchant_risk"
]

X = df[features]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

false_positive_cost = fp * 50

# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🛡️ RazorShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Payment Fraud Risk Detection & Decision Engine'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# PERFORMANCE
# ==================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4, m5 = st.columns(5)

metrics = [
    ("Accuracy", f"{accuracy:.2%}"),
    ("Precision", f"{precision:.2%}"),
    ("Recall", f"{recall:.2%}"),
    ("F1 Score", f"{f1:.2%}"),
    ("FP Cost", f"₹{false_positive_cost:,}")
]

for col, (title, value) in zip(
    [m1, m2, m3, m4, m5],
    metrics
):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()

# ==================================================
# SIDEBAR INPUT
# ==================================================

st.sidebar.title("🔎 Transaction Analysis")
st.sidebar.caption("Enter transaction signals")

amount = st.sidebar.number_input(
    "Transaction Amount (₹)",
    min_value=1.0,
    value=25000.0
)

transaction_hour = st.sidebar.slider(
    "Transaction Hour",
    0,
    23,
    2
)

customer_age = st.sidebar.slider(
    "Customer Age",
    18,
    80,
    25
)

transactions_last_24h = st.sidebar.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    value=10
)

avg_transaction_amount = st.sidebar.number_input(
    "Average Transaction Amount (₹)",
    min_value=1.0,
    value=2000.0
)

distance_from_last_transaction = st.sidebar.number_input(
    "Distance From Last Transaction (km)",
    min_value=0.0,
    value=150.0
)

is_new_device = st.sidebar.selectbox(
    "New Device?",
    ["No", "Yes"]
)

is_international = st.sidebar.selectbox(
    "International Transaction?",
    ["No", "Yes"]
)

failed_attempts = st.sidebar.number_input(
    "Failed Attempts",
    min_value=0,
    value=4
)

merchant_risk = st.sidebar.slider(
    "Merchant Risk Level",
    1,
    5,
    5
)

analyze = st.sidebar.button(
    "🔍 Analyze Transaction",
    use_container_width=True
)

# ==================================================
# INPUT DATA
# ==================================================

input_data = pd.DataFrame([{
    "amount": amount,
    "transaction_hour": transaction_hour,
    "customer_age": customer_age,
    "transactions_last_24h": transactions_last_24h,
    "avg_transaction_amount": avg_transaction_amount,
    "distance_from_last_transaction": distance_from_last_transaction,
    "is_new_device": 1 if is_new_device == "Yes" else 0,
    "is_international": 1 if is_international == "Yes" else 0,
    "failed_attempts": failed_attempts,
    "merchant_risk": merchant_risk
}])

# ==================================================
# DEFAULT SCREEN
# ==================================================

if not analyze:

    st.markdown(
        '<div class="section-title">🚀 AI-Powered Transaction Screening</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            '<div class="info-box"><b>1️⃣ Analyze</b><br><br>'
            'Evaluate transaction amount, device, location, frequency and merchant signals.'
            '</div>',
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            '<div class="info-box"><b>2️⃣ Score</b><br><br>'
            'Machine learning estimates the probability of fraudulent activity.'
            '</div>',
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            '<div class="info-box"><b>3️⃣ Decide</b><br><br>'
            'Convert risk into an actionable payment decision.'
            '</div>',
            unsafe_allow_html=True
        )

    st.info(
        "👈 Enter transaction details in the left panel and click "
        "**Analyze Transaction**."
    )

# ==================================================
# ANALYSIS
# ==================================================

if analyze:

    probability = model.predict_proba(input_data)[0][1]
    risk_score = probability * 100

    if risk_score < 30:
        risk_level = "LOW"
        action = "APPROVE"
    elif risk_score < 70:
        risk_level = "MEDIUM"
        action = "ADDITIONAL VERIFICATION"
    else:
        risk_level = "HIGH"
        action = "MANUAL REVIEW"

    # ------------------------------------------------
    # RESULTS
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 AI Risk Assessment</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Fraud Probability</div>
                <div class="result-value">{probability:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Risk Score</div>
                <div class="result-value">{risk_score:.1f}/100</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r3:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Risk Level</div>
                <div class="result-value">{risk_level}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(min(int(risk_score), 100))

    if risk_level == "LOW":
        st.success(f"🟢 **Recommended Action: {action}**")
    elif risk_level == "MEDIUM":
        st.warning(f"🟡 **Recommended Action: {action}**")
    else:
        st.error(f"🔴 **Recommended Action: {action}**")

    # ------------------------------------------------
    # EXPLANATION
    # ------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">🔍 Explainable AI — Why is this transaction risky?</div>',
        unsafe_allow_html=True
    )

    reasons = []

    if amount > avg_transaction_amount * 4:
        reasons.append(
            "💰 Transaction amount is significantly higher than the customer's average."
        )

    if is_new_device == "Yes":
        reasons.append("📱 New device detected.")

    if is_international == "Yes":
        reasons.append("🌍 International transaction detected.")

    if failed_attempts >= 3:
        reasons.append("⚠️ Multiple failed attempts detected.")

    if distance_from_last_transaction > 100:
        reasons.append("📍 Unusually large distance from the previous transaction.")

    if transactions_last_24h > 8:
        reasons.append("🔄 High transaction frequency detected.")

    if merchant_risk >= 4:
        reasons.append("🏪 High merchant risk level detected.")

    if transaction_hour <= 4:
        reasons.append("🌙 Transaction occurred during an unusual late-night period.")

    if not reasons:
        reasons.append("✅ No major suspicious indicators detected.")

    for reason in reasons:
        st.write(reason)

    # ------------------------------------------------
    # DECISION SUMMARY
    # ------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">🤖 AI Decision Summary</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"Risk Score: **{risk_score:.1f}/100**  |  "
        f"Risk Level: **{risk_level}**  |  "
        f"Action: **{action}**"
    )

st.markdown(
    '<div class="footer">'
    'RazorShield AI • Defensive Fraud Risk Prototype • Synthetic Transaction Data'
    '</div>',
    unsafe_allow_html=True
)
