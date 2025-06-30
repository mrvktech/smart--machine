# 🤖 Smart Manufacturing Machines Efficiency Prediction

## 📌 Problem Statement
In modern manufacturing environments, machine inefficiencies result in production losses, energy waste, and increased maintenance costs. There is a need for an intelligent system that can predict machine efficiency levels in real-time using operational sensor data.

---

## 🎯 Objective
To develop a machine learning-based classification system that predicts the **Efficiency Status** (High, Medium, Low) of manufacturing machines, aiding in predictive maintenance and operational optimization.

---

## 💡 Proposed System Features
- Data cleaning, encoding, and scaling
- Feature selection using Random Forest and ANOVA
- Multi-model training: Logistic Regression, SVM, XGBoost, Random Forest
- Evaluation using accuracy, confusion matrix, and ROC curve
- Web deployment using Streamlit for real-time predictions
- Models saved using `joblib`

---

## 🛠️ Tools & Technologies Used
- **Programming Language**: Python
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`
- **Web App**: Streamlit

---

## 📁 Dataset
- **Source**: [Kaggle - Intelligent Manufacturing Dataset](https://www.kaggle.com/datasets/ziya07/intelligent-manufacturing-dataset)
- **Rows**: 100,000+
- **Target Variable**: `Efficiency_Status`  
- **Features**: Sensor readings like temperature, vibration, power consumption, error rate, etc.

---

## ✅ Selected Features
Features selected using Random Forest importance & ANOVA:
- `Production_Speed_units_per_hr`
- `Error_Rate_%`

---

## 🔍 Models Trained
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost Classifier
- Random Forest Classifier

---

## 📊 Evaluation Metrics
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix (Visualized)
- ROC Curve (One-vs-Rest)

---

## 💻 Streamlit Web App
Features of the deployed app:
- Input form to enter machine metrics
- Real-time prediction of efficiency class
- Visualization of Confusion Matrix & ROC Curve
- Backend models loaded from saved `.pkl` files

---

## 📷 UI & Output Snapshots
- Streamlit Input Form  
- Model Output: Predicted Efficiency Status  
- Evaluation Charts: Confusion Matrix & ROC Curve

---

## 🚀 Future Scope
- Integrate with live sensor data from IoT devices
- Add time-series forecasting for trend prediction
- Build predictive maintenance alerts
- Centralized dashboard for factory-wide monitoring

---

## 👥 Team Members
- Vishwajeet Kumar
- Vishal Kumar Mahto
- Ankit Kumar
- Kritika Kumari

---

## 📬 Contact
For any queries or collaborations, reach out at:  
**mailme.wivk@gmail.com**

