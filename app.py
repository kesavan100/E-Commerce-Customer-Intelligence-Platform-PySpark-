import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Customer Analytics App",
    page_icon="📊",
    layout="centered"
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
# UI
# -----------------------------------
st.title("📊 Customer Analytics App")

prediction_type = st.selectbox(
    "Choose what you want to find",
    ["Churn Prediction", "Customer Segmentation", "Engagement Score"]
)

st.subheader("Enter Customer Details")

user_id = st.text_input("User ID", value="82932")
purchases = st.number_input("Purchases", min_value=0.0, value=10.0, step=1.0)
total_spent = st.number_input("Total Spent", min_value=0.0, value=5000.0, step=1.0)
avg_purchase_price = st.number_input("Average Purchase Price", min_value=0.0, value=500.0, step=1.0)
product_diversity = st.number_input("Product Diversity", min_value=0.0, value=6.0, step=1.0)
category_diversity = st.number_input("Category Diversity", min_value=0.0, value=3.0, step=1.0)
session_count = st.number_input("Session Count", min_value=0.0, value=20.0, step=1.0)

if st.button("Predict"):
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

            st.success(f"Prediction completed for User ID: {user_id}")

            if prediction_type == "Churn Prediction":
                probability, label = predict_churn(input_df)

                st.write(f"### Churn Probability: {probability:.2f}%")
                st.info(f"Prediction: {label}")

            elif prediction_type == "Customer Segmentation":
                cluster, segment_label = predict_segmentation(input_df)

                st.write(f"### Customer Segment: {segment_label}")
                st.info(f"Cluster ID: {cluster}")

            elif prediction_type == "Engagement Score":
                engagement_score, cluster, engagement_label = predict_engagement(input_df)

                st.write(f"### Engagement Score: {engagement_score:.2f}")
                st.write(f"### Engagement Level: {engagement_label}")
                st.info(f"Cluster ID: {cluster}")

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Make sure all .pkl files are in the same folder as this app.py file.")