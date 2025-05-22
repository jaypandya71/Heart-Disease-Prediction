import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("heart.csv")
X = data.drop(columns=["target"], axis=1)
y = data["target"]

# Train-test split and SMOTE balancing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'heart_disease_model.pkl')

# Streamlit frontend UI
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f8cdda, #1d2a6c);
        font-family: 'Arial', sans-serif;
        color: #fff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
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
        grid-template-columns: repeat(4, 1fr);
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
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💓 Heart Disease Prediction App 💓")

st.header("📊 Heart Disease Dataset Overview")
st.dataframe(data.head())

st.header("📈 Data Statistics")
st.write(data.describe())

CLASSIFIED_DATA_FILE = "classified_data.csv"
if os.path.exists(CLASSIFIED_DATA_FILE):
    classified_data = pd.read_csv(CLASSIFIED_DATA_FILE)
else:
    classified_data = pd.DataFrame(columns=data.columns)

if 'classified_data' not in st.session_state:
    st.session_state.classified_data = classified_data

if 'learning_curves_fig' not in st.session_state:
    st.session_state.learning_curves_fig = None

if 'roc_curve_fig' not in st.session_state:
    st.session_state.roc_curve_fig = None

if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None

st.header("Provide Your Health Information Below")

def user_input_features():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.slider("Age", 29, 71, 50)
    with col2:
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    with col3:
        sex = st.radio("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    with col4:
        chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 200)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cp = st.selectbox("Chest Pain Type (0-3)", list(range(4)))
    with col2:
        restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    with col3:
        thalach = st.slider("Max Heart Rate Achieved", 71, 202, 140)
    with col4:
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        exang = st.radio("Exercise-Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.2, 1.0)
    with col3:
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    with col4:
        ca = st.slider("Number of Major Vessels", 0, 3, 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        thal = st.selectbox("Thalassemia (0, 1, 2, 3)", [0, 1, 2, 3])
    st.markdown('</div>', unsafe_allow_html=True)

    features = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()
st.subheader("User Input Features")
st.write(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Results")
    st.markdown('<div class="output-container">', unsafe_allow_html=True)
    if prediction[0] == 1:
        st.error("🚫 Prediction: You might have a heart disease. Stay strong and consult a doctor.")
    else:
        st.success("😊 Prediction: No signs of heart disease. Keep up the healthy lifestyle!")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📊 Prediction Confidence")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"❤️ Probability of Heart Disease: {prediction_proba[0][1]:.2%}")
        st.write(f"💚 Probability of No Heart Disease: {prediction_proba[0][0]:.2%}")
    with col2:
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        st.write(f"🎯 Model Accuracy: {test_accuracy:.2%}")

    input_df['target'] = prediction[0]
    st.session_state.classified_data = pd.concat([st.session_state.classified_data, input_df], ignore_index=True)
    st.session_state.classified_data.to_csv(CLASSIFIED_DATA_FILE, index=False)

    st.subheader("All Classified Data")
    st.write(st.session_state.classified_data)

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    }, index=['No Heart Disease', 'Heart Disease'])
    for col in ['Precision', 'Recall', 'F1-Score']:
        metrics_df[col] = metrics_df[col].map('{:.2%}'.format)
    st.session_state.evaluation_metrics = metrics_df

# 📈 ROC Curve
st.subheader("📈 ROC Curve Analysis")
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_roc = roc_auc_score(y_test, y_pred_proba)

def plot_roc_curve():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_roc:.2f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True)
    return fig

if st.session_state.roc_curve_fig is None:
    with st.spinner("Generating ROC curve..."):
        st.session_state.roc_curve_fig = plot_roc_curve()
st.pyplot(st.session_state.roc_curve_fig)

with st.expander("ℹ️ Understanding the ROC Curve"):
    st.markdown("""
    The ROC Curve shows the trade-off between **True Positive Rate** and **False Positive Rate** at different thresholds.

    - **AUC (Area Under Curve)**: Closer to **1.0** indicates better model performance.
    - A higher curve means better **discrimination ability** between classes.
    """)

# 📊 Learning Curves
st.subheader("📊 Model Learning Curves")
def plot_learning_curves():
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_mean, val_mean, train_std, val_std = [], [], [], []
    for size in train_sizes:
        size = int(size * len(X_train))
        X_sub = X_train[:size]
        y_sub = y_train[:size]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        t_scores, v_scores = [], []
        for tr, vl in kf.split(X_sub):
            X_tr, y_tr = X_sub.iloc[tr], y_sub.iloc[tr]
            X_vl, y_vl = X_sub.iloc[vl], y_sub.iloc[vl]
            m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              use_label_encoder=False, eval_metric='logloss', random_state=42)
            m.fit(X_tr, y_tr)
            t_scores.append(m.score(X_tr, y_tr))
            v_scores.append(m.score(X_vl, y_vl))
        train_mean.append(np.mean(t_scores))
        val_mean.append(np.mean(v_scores))
        train_std.append(np.std(t_scores))
        val_std.append(np.std(v_scores))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.fill_between(train_sizes, np.array(train_mean)-train_std, np.array(train_mean)+train_std, alpha=0.1)
    ax1.fill_between(train_sizes, np.array(val_mean)-val_std, np.array(val_mean)+val_std, alpha=0.1)
    ax1.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
    ax1.plot(train_sizes, val_mean, 'o-', label="Validation Accuracy")
    ax1.set_title("Model Accuracy vs Training Size")
    ax1.set_xlabel("Training Size")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_sizes, [1 - x for x in train_mean], 'o-', label="Training Error")
    ax2.plot(train_sizes, [1 - x for x in val_mean], 'o-', label="Validation Error")
    ax2.set_title("Model Error vs Training Size")
    ax2.set_xlabel("Training Size")
    ax2.set_ylabel("Error Rate")
    ax2.legend()
    ax2.grid(True)

    return fig

if st.session_state.learning_curves_fig is None:
    with st.spinner("Generating learning curves..."):
        st.session_state.learning_curves_fig = plot_learning_curves()
st.pyplot(st.session_state.learning_curves_fig)

with st.expander("ℹ️ Understanding the Learning Curves"):
    st.markdown("""
    **Left Plot (Accuracy):**
    - **Higher values are better**
    - Shows how accurately the model predicts both training and validation data
    - **Converging lines** indicate a good model fit

    **Right Plot (Error):**
    - **Lower values are better**
    - Shows the model's error rate on training and validation data
    - Helps identify **overfitting or underfitting**

    **What to Look For:**
    - Small gap between training and validation curves
    - Stable performance as training size increases
    - Low variance (narrow shaded areas)
    """)

# 📋 Metrics
st.subheader("📋 Model Performance Metrics")

if st.session_state.evaluation_metrics is not None:
    st.table(st.session_state.evaluation_metrics)

    with st.expander("ℹ️ Understanding the Performance Metrics"):
        st.markdown("""
        ### 📊 Metric Definitions

        - **Accuracy**: Overall, how often is the classifier correct?  
        *(Accuracy = (TP + TN) / (TP + TN + FP + FN))*

        - **Precision**: When the model predicts **heart disease**, how often is it correct?  
        *(Precision = TP / (TP + FP))*

        - **Recall (Sensitivity)**: Out of all **actual heart disease cases**, how many did the model identify?  
        *(Recall = TP / (TP + FN))*

        - **F1-Score**: A harmonic mean of Precision and Recall. Useful when you need a balance between both.  
        *(F1 = 2 × (Precision × Recall) / (Precision + Recall))*

        - **Support**: Number of samples of the true response that lie in each class.

        - **AUC-ROC (if available)**: Measures the model’s ability to distinguish between classes.  
        A higher AUC means better performance across all classification thresholds.
        """)



