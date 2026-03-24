import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Customer Analytics App",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# Load saved sklearn models
# -----------------------------------
@st.cache_resource
def load_artifacts():
    churn_model = joblib.load("churn_rf_model1.pkl")
    churn_feature_cols = joblib.load("churn_feature_cols1.pkl")

    segment_scaler = joblib.load("segment_scaler.pkl")
    segment_model = joblib.load("segment_model.pkl")
    segment_feature_cols = joblib.load("segment_feature_cols.pkl")

    engagement_scaler = joblib.load("engagement_scaler.pkl")
    engagement_model = joblib.load("engagement_model.pkl")
    engagement_label_map = joblib.load("engagement_label_map.pkl")

    return (
        churn_model,
        churn_feature_cols,
        segment_scaler,
        segment_model,
        segment_feature_cols,
        engagement_scaler,
        engagement_model,
        engagement_label_map,
    )

(
    churn_model,
    churn_feature_cols,
    segment_scaler,
    segment_model,
    segment_feature_cols,
    engagement_scaler,
    engagement_model,
    engagement_label_map,
) = load_artifacts()

# -----------------------------------
# Helper functions
# -----------------------------------
def make_input_dataframe(
    purchases,
    total_spent,
    avg_purchase_price,
    product_diversity,
    category_diversity,
    session_count
):
    return pd.DataFrame([{
        "purchases": float(purchases),
        "total_spent": float(total_spent),
        "avg_purchase_price": float(avg_purchase_price),
        "product_diversity": float(product_diversity),
        "category_diversity": float(category_diversity),
        "session_count": float(session_count),
    }])

def predict_churn(input_df):
    input_df = input_df[churn_feature_cols]
    prediction = churn_model.predict(input_df)[0]
    probability = churn_model.predict_proba(input_df)[0][1] * 100
    label = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"
    return probability, label

def predict_segmentation(input_df):
    input_scaled = segment_scaler.transform(input_df[segment_feature_cols])
    cluster = segment_model.predict(input_scaled)[0]

    segment_map = {
        2: "High-Value",
        0: "Regular",
        1: "At-Risk"
    }

    segment_label = segment_map.get(cluster, f"Cluster {cluster}")
    return cluster, segment_label

def predict_engagement(input_df):
    input_df = input_df.copy()

    input_df["engagement_score"] = (
        0.4 * input_df["purchases"] +
        0.3 * input_df["session_count"] +
        0.2 * input_df["product_diversity"] +
        0.1 * input_df["category_diversity"]
    )

    scaled_score = engagement_scaler.transform(input_df[["engagement_score"]])
    cluster = engagement_model.predict(scaled_score)[0]
    engagement_label = engagement_label_map.get(cluster, f"Cluster {cluster}")

    return float(input_df["engagement_score"].iloc[0]), cluster, engagement_label

# -----------------------------------
# Custom CSS
# -----------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-box {
        background: rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 24px;
        padding: 28px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.20);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        color: #dbeafe;
        font-size: 1.02rem;
        line-height: 1.6;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.11);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        margin-bottom: 22px;
    }

    .section-title {
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .result-box {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        border-radius: 20px;
        padding: 24px;
        color: white;
        box-shadow: 0 12px 35px rgba(37, 99, 235, 0.35);
        margin-top: 20px;
    }

    .result-box h3 {
        margin: 0 0 8px 0;
        font-size: 1.4rem;
        color: white;
    }

    .result-box p {
        margin: 8px 0;
        font-size: 1rem;
        color: white;
    }

    .mini-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 18px;
        padding: 14px 16px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }

    .mini-title {
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    .mini-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
    }

    label, .stMarkdown, .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: white !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.95) !important;
        border: none !important;
        border-radius: 16px !important;
        min-height: 54px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12) !important;
    }

    div[data-baseweb="input"] > div {
        background: rgba(255,255,255,0.96) !important;
        border: none !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12) !important;
        min-height: 52px !important;
    }

    div[data-baseweb="input"] input {
        color: #0f172a !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding-left: 6px !important;
    }

    .stNumberInput button {
        border-radius: 12px !important;
    }

    .stButton > button {
        width: 100%;
        height: 54px;
        border: none;
        border-radius: 16px;
        background: linear-gradient(90deg, #38bdf8, #6366f1, #8b5cf6);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(99,102,241,0.35);
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(99,102,241,0.45);
    }

    .footer-note {
        text-align: center;
        color: #cbd5e1;
        font-size: 0.92rem;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Header
# -----------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">📊 Customer Analytics Dashboard</div>
    <div class="hero-subtitle">
        Run churn prediction, customer segmentation, and engagement analysis
        with a cleaner and more premium interface.
    </div>
</div>
""", unsafe_allow_html=True)

# top stats look
a, b, c = st.columns(3)
with a:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-title">Models Available</div>
        <div class="mini-value">3</div>
    </div>
    """, unsafe_allow_html=True)
with b:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-title">Analysis Types</div>
        <div class="mini-value">Churn / Segment / Engagement</div>
    </div>
    """, unsafe_allow_html=True)
with c:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-title">Input Mode</div>
        <div class="mini-value">Manual Customer Data</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# Analysis type
# -----------------------------------
st.markdown('<div class="section-title">Choose Analysis Type</div>', unsafe_allow_html=True)

prediction_type = st.selectbox(
    "",
    ["Churn Prediction", "Customer Segmentation", "Engagement Score"],
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Input section
# -----------------------------------
st.markdown('<div class="section-title">Enter Customer Details</div>', unsafe_allow_html=True)

user_id = st.text_input("User ID", value="82932", placeholder="Enter customer ID")

col1, col2, col3 = st.columns(3)

with col1:
    purchases = st.number_input("Purchases", min_value=0.0, value=10.0, step=1.0)
    total_spent = st.number_input("Total Spent", min_value=0.0, value=5000.0, step=1.0)

with col2:
    avg_purchase_price = st.number_input("Average Purchase Price", min_value=0.0, value=500.0, step=1.0)
    product_diversity = st.number_input("Product Diversity", min_value=0.0, value=6.0, step=1.0)

with col3:
    category_diversity = st.number_input("Category Diversity", min_value=0.0, value=3.0, step=1.0)
    session_count = st.number_input("Session Count", min_value=0.0, value=20.0, step=1.0)

predict_btn = st.button("🚀 Run Analysis")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# Output
# -----------------------------------
if predict_btn:
    try:
        if not user_id.strip():
            st.warning("Please enter User ID.")
        else:
            input_df = make_input_dataframe(
                purchases=purchases,
                total_spent=total_spent,
                avg_purchase_price=avg_purchase_price,
                product_diversity=product_diversity,
                category_diversity=category_diversity,
                session_count=session_count,
            )

            if prediction_type == "Churn Prediction":
                probability, label = predict_churn(input_df)

                st.markdown(f"""
                <div class="result-box">
                    <h3>Churn Prediction Result</h3>
                    <p><b>User ID:</b> {user_id}</p>
                    <p><b>Churn Probability:</b> {probability:.2f}%</p>
                    <p><b>Prediction:</b> {label}</p>
                </div>
                """, unsafe_allow_html=True)

            elif prediction_type == "Customer Segmentation":
                cluster, segment_label = predict_segmentation(input_df)

                st.markdown(f"""
                <div class="result-box">
                    <h3>Customer Segmentation Result</h3>
                    <p><b>User ID:</b> {user_id}</p>
                    <p><b>Segment:</b> {segment_label}</p>
                    <p><b>Cluster ID:</b> {cluster}</p>
                </div>
                """, unsafe_allow_html=True)

            elif prediction_type == "Engagement Score":
                engagement_score, cluster, engagement_label = predict_engagement(input_df)

                st.markdown(f"""
                <div class="result-box">
                    <h3>Engagement Analysis Result</h3>
                    <p><b>User ID:</b> {user_id}</p>
                    <p><b>Engagement Score:</b> {engagement_score:.2f}</p>
                    <p><b>Engagement Level:</b> {engagement_label}</p>
                    <p><b>Cluster ID:</b> {cluster}</p>
                </div>
                """, unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown(
    """
    <div class="footer-note">
        ⚙️ Data processed using PySpark (Big Data Pipeline) <br>
        🧠 Models trained using both PySpark ML and scikit-learn <br>
        🚀 Deployed using scikit-learn for fast, lightweight Streamlit inference <br><br>
    </div>
    """,
    unsafe_allow_html=True
)