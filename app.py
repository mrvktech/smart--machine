import streamlit as st
import joblib
import numpy as np

# Load models
model_names = {
    'XGBoost': 'models/xgb_best_model.pkl',
    'SVM': 'models/svm_best_model.pkl',
    'Logistic Regression': 'models/logreg_best_model.pkl',
    'Random Forest' : 'models/rf_best_model.pkl'
}

# Load label encoder
label_encoder = joblib.load('models/label_encoder.pkl')

# Sidebar model selection
st.sidebar.title("🔍 Choose Model")
model_choice = st.sidebar.selectbox("Select trained model", list(model_names.keys()))

# Load selected model
model = joblib.load(model_names[model_choice])

st.title("🛠️ Smart Manufacturing: Efficiency Status Predictor")

# Input feature form
st.markdown("### 📥 Enter Machine Sensor Data")

# Replace this with your actual selected features
input_features = [
    'Production_Speed_units_per_hr',
    'Error_Rate_%'
]

user_input = {}
for feature in input_features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to array for prediction
input_array = np.array([list(user_input.values())]).reshape(1, -1)

# Predict
if st.button("🔮 Predict Efficiency Status"):
    prediction = model.predict(input_array)
    pred_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"✅ Predicted Efficiency Status: **{pred_label}**")
