import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart.csv")

# Split data into features and target
X = data.drop(columns=["target"], axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Inject custom CSS styles
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f8cdda, #1d2a6c);
        font-family: 'Arial', sans-serif;
        color: #fff;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
    }
    .stButton>button:hover {
        background-color: #ff4d4d;
    }
    .stMarkdown, .stRadio, .stSlider, .stSelectbox {
        padding: 15px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .input-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);  /* 4 columns per row */
        gap: 15px;
        margin-bottom: 30px;
    }
    .input-container > div {
        display: flex;
        flex-direction: column;
    }
    .stSubheader {
        color: #ffecd2;
    }
    .stWrite {
        color: #dcdcdc;
    }
    .output-container {
        display: grid;
        grid-template-columns: 1fr 1fr; /* Two columns for output */
        gap: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("💓 Heart Disease Prediction App 💓")

# Display a sample of the dataset
st.header("📊 Heart Disease Dataset Overview")
st.write("Here is a sample of the dataset used for prediction:")
st.dataframe(data.head())

# Input fields for user to provide values
st.header("Provide Your Health Information Below")

# Arrange inputs in a grid-based layout (4 columns)
def user_input_features():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.slider("Age", min_value=29, max_value=71, value=50, step=1)
    with col2:
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=94, max_value=200, value=120, step=1)
    with col3:
        sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    with col4:
        chol = st.slider("Serum Cholesterol (mg/dl)", min_value=126, max_value=564, value=200, step=1)

    # Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cp = st.selectbox("Chest Pain Type (0-3)", options=list(range(4)))
    with col2:
        restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
    with col3:
        thalach = st.slider("Max Heart Rate Achieved", min_value=71, max_value=202, value=140, step=1)
    with col4:
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Row 3
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        exang = st.radio("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        oldpeak = st.slider("Oldpeak (ST Depression)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    with col3:
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    with col4:
        ca = st.slider("Number of Major Vessels", min_value=0, max_value=3, value=0)

    # Row 4
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        thal = st.selectbox("Thalassemia (0, 1, 2, 3)", options=[0, 1, 2, 3])

    st.markdown('</div>', unsafe_allow_html=True)

    features = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return pd.DataFrame(features, index=[0])

# Get user input
input_df = user_input_features()

# Display the user inputs (side by side)
st.subheader("User Input Features")
st.write(input_df)

# Predict and display results
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display prediction results in a grid layout
st.subheader("Prediction Results")
st.markdown('<div class="output-container">', unsafe_allow_html=True)

if prediction[0] == 1:
    st.error("🚫 Prediction: You might have a heart disease. Stay strong and consult a doctor.")
else:
    st.success("😊 Prediction: No signs of heart disease. Keep up the healthy lifestyle!")

st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Prediction Probability")
st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
