import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease AI", layout="wide")


model_data = joblib.load("heart_model.pkl")
model = model_data["model"]
features = model_data["features"]


st.markdown("<h1 style='text-align:center;'>❤️ AI-Based Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")


left_col, right_col = st.columns([1, 2])


with left_col:
    st.subheader("📝 Enter Patient Details")

    input_data = {}

    for feature in features:
        input_data[feature] = st.number_input(
            f"{feature}",
            value=0.0
        )

    predict_button = st.button("🔍 Predict Risk")

    st.markdown("---")
    st.markdown("✅ Built using:")
    st.markdown("- Scikit-learn Pipeline")
    st.markdown("- Cross Validation")
    st.markdown("- Random Forest")
    st.markdown("- Streamlit UI")


with right_col:

    if predict_button:

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:
            st.error(f"""
            ⚠️ **High Risk of Heart Disease**

            Probability: **{probability*100:.2f}%**
            """)
        else:
            st.success(f"""
            ✅ **Low Risk of Heart Disease**

            Probability: **{probability*100:.2f}%**
            """) 

    else:
        st.info("Enter patient data on the left and click Predict.")