import streamlit as st
import joblib
import pandas as pd

# Saved model, scaler, columns load karo
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("📊 Customer Churn Prediction")
st.write("Customer ki details daal kar pata karo wo churn karega ya nahi")

# User inputs lo
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

if st.button("Predict Churn"):
    # Step 1: Raw input ko dictionary mein daalo (training data jaisa format)
    input_dict = {
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_dict])

    # Step 2: Same one-hot encoding training jaisi
    cat_cols = input_df.select_dtypes(include='object').columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=cat_cols)

    # Step 3: Missing columns add karo (jo training mein thi but yahan nahi banin)
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Step 4: Sahi order mein columns arrange karo
    input_encoded = input_encoded[model_columns]

    # Step 5: Scale karo
    input_scaled = scaler.transform(input_encoded)

    # Step 6: Predict karo
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Ye customer CHURN kar sakta hai! (Probability: {probability:.1%})")
    else:
        st.success(f"✅ Ye customer STAY karega. (Churn Probability: {probability:.1%})")