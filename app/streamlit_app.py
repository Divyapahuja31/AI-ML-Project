import streamlit as st
import pandas as pd
import joblib
import os


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "risk_model.pkl"
)


st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="centered"
)


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Unable to load model: {e}")
        return None


def main():
    st.title("🏦 AI Credit Risk Scoring System")
    st.write(
        "Enter borrower details to estimate the probability "
        "of credit delinquency within the next two years."
    )

    model = load_model()

    if model is None:
        st.stop()

    tab1, tab2 = st.tabs(["Manual Entry", "File Upload (CSV)"])

    with tab1:
        with st.form("risk_form"):
            st.subheader("Applicant Details")
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 18, 120, 35)
                monthly_inc = st.number_input("Monthly Income", 0, 100000, 5000)
                rev_util = st.number_input("Credit Utilization", 0.0, 20000.0, 0.5)
                debt_ratio = st.number_input("Debt Ratio", 0.0, 10.0, 0.35)
                dependents = st.number_input("Dependents", 0, 20, 0)

            with col2:
                open_credit = st.number_input("Open Credit Lines", 0, 50, 5)
                real_estate = st.number_input("Real Estate Loans", 0, 10, 1)
                late_30_59 = st.number_input("30–59 Days Late", 0, 20, 0)
                late_60_89 = st.number_input("60–89 Days Late", 0, 20, 0)
                late_90 = st.number_input("90+ Days Late", 0, 20, 0)

            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            input_data = pd.DataFrame([{
                "rev_util": rev_util, "age": age, "late_30_59": late_30_59, "debt_ratio": debt_ratio,
                "monthly_inc": monthly_inc, "open_credit": open_credit, "late_90": late_90,
                "real_estate": real_estate, "late_60_89": late_60_89, "dependents": dependents
            }])

            with st.spinner("Evaluating credit profile..."):
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("High Risk Applicant")
            else:
                st.success("Low Risk Applicant")

            st.write(f"Estimated default probability: **{probability*100:.2f}%**")

            with st.expander("Model Insight"):
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": input_data.columns,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)
                st.dataframe(importance_df.head(5), use_container_width=True)

    with tab2:
        st.subheader("Upload Borrower Data")
        st.markdown("Upload a CSV file containing borrower rows to receive batch predictions.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(batch_data.head())
            
            required_cols = ["rev_util", "age", "late_30_59", "debt_ratio", "monthly_inc", 
                             "open_credit", "late_90", "real_estate", "late_60_89", "dependents"]
                             
            if all(col in batch_data.columns for col in required_cols):
                if st.button("Predict Batch Risk"):
                    with st.spinner("Processing batch predictions..."):
                        X_batch = batch_data[required_cols].fillna(batch_data[required_cols].median())
                        preds = model.predict(X_batch)
                        probs = model.predict_proba(X_batch)[:, 1]
                        
                        batch_data["Risk_Prediction"] = ["High Risk" if p == 1 else "Low Risk" for p in preds]
                        batch_data["Default_Probability"] = probs
                        
                        st.success("Batch Prediction Complete!")
                        st.dataframe(batch_data[["Risk_Prediction", "Default_Probability"] + required_cols])
            else:
                st.error(f"Error: the uploaded CSV does not contain the correct features. Expected features: {required_cols}")


if __name__ == "__main__":
    main()