import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os

# Load dataset
data = pd.read_csv("heart.csv")

# Split data into features and target
X = data.drop(columns=["target"], axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'heart_disease_model.pkl')

# Inject custom CSS styles
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f8cdda, #1d2a6c);
        font-family: 'Arial', sans-serif;
        color: #fff;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
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

# App title
st.title("💓 Heart Disease Prediction App 💓")

# Display a sample of the dataset
st.header("📊 Heart Disease Dataset Overview")
st.write("Here is a sample of the dataset used for prediction:")
st.dataframe(data.head())

# Display statistics metrics of the data
st.header("📈 Data Statistics")
st.write("Summary statistics of the dataset:")
st.write(data.describe())

# File to store classified data
CLASSIFIED_DATA_FILE = "classified_data.csv"

# Load classified data from file if it exists
if os.path.exists(CLASSIFIED_DATA_FILE):
    classified_data = pd.read_csv(CLASSIFIED_DATA_FILE)
else:
    classified_data = pd.DataFrame(columns=data.columns)

# Initialize session state for storing classified data
if 'classified_data' not in st.session_state:
    st.session_state.classified_data = classified_data

# Initialize session state for storing graphs
if 'learning_curves_fig' not in st.session_state:
    st.session_state.learning_curves_fig = None

if 'roc_curve_fig' not in st.session_state:
    st.session_state.roc_curve_fig = None

# Initialize session state for storing evaluation metrics
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None

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
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame(features, index=[0])

# Get user input
input_df = user_input_features()

# Display the user inputs
st.subheader("User Input Features")
st.write(input_df)

# Button to trigger prediction
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display prediction results
    st.subheader("Prediction Results")
    st.markdown('<div class="output-container">', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("🚫 Prediction: You might have a heart disease. Stay strong and consult a doctor.")
    else:
        st.success("😊 Prediction: No signs of heart disease. Keep up the healthy lifestyle!")

    st.markdown('</div>', unsafe_allow_html=True)

    # Show prediction probabilities and model accuracy
    st.subheader("📊 Prediction Confidence")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Prediction Probabilities:")
        st.write(f"❤️ Probability of Heart Disease: {prediction_proba[0][1]:.2%}")
        st.write(f"💚 Probability of No Heart Disease: {prediction_proba[0][0]:.2%}")

    with col2:
        # Calculate and display accuracy
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        st.write("Model Accuracy:")
        st.write(f"🎯 The model's prediction accuracy is: {test_accuracy:.2%}")

    # Append new classified data to session state
    input_df['target'] = prediction[0]
    st.session_state.classified_data = pd.concat([st.session_state.classified_data, input_df], ignore_index=True)

    # Save classified data to file
    st.session_state.classified_data.to_csv(CLASSIFIED_DATA_FILE, index=False)

    # Display all classified data
    st.subheader("All Classified Data")
    st.write(st.session_state.classified_data)

    # Store evaluation metrics in session state
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

# Learning Curves
st.subheader("📊 Model Learning Curves")
st.markdown("""
This visualization shows how the model's performance improves with more training data:
- **Blue lines**: Training performance
- **Orange lines**: Validation performance
- **Shaded areas**: Performance variance
""")

def plot_learning_curves():
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_mean = []
    val_mean = []
    train_std = []
    val_std = []

    for train_size in train_sizes:
        train_size = int(len(X_train) * train_size)
        X_subset = X_train[:train_size]
        y_subset = y_train[:train_size]
        
        cv_scores_train = []
        cv_scores_val = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X_subset):
            X_train_cv = X_subset.iloc[train_idx]
            y_train_cv = y_subset.iloc[train_idx]
            X_val_cv = X_subset.iloc[val_idx]
            y_val_cv = y_subset.iloc[val_idx]
            
            model_cv = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model_cv.fit(X_train_cv, y_train_cv)
            
            train_score = model_cv.score(X_train_cv, y_train_cv)
            val_score = model_cv.score(X_val_cv, y_val_cv)
            
            cv_scores_train.append(train_score)
            cv_scores_val.append(val_score)
        
        train_mean.append(np.mean(cv_scores_train))
        train_std.append(np.std(cv_scores_train))
        val_mean.append(np.mean(cv_scores_val))
        val_std.append(np.std(cv_scores_val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Plot accuracy curves
    ax1.fill_between(train_sizes, 
                    np.array(train_mean) - np.array(train_std),
                    np.array(train_mean) + np.array(train_std), 
                    alpha=0.1, color='blue', label='Training Variance')
    ax1.fill_between(train_sizes, 
                    np.array(val_mean) - np.array(val_std),
                    np.array(val_mean) + np.array(val_std), 
                    alpha=0.1, color='orange', label='Validation Variance')
    ax1.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy', linewidth=2)
    ax1.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy vs Training Size', fontsize=14, pad=20)
    ax1.set_xlabel('Number of Training Examples', fontsize=12)
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim([0.6, 1.0])

    # Plot error curves
    ax2.fill_between(train_sizes, 
                    1 - (np.array(train_mean) - np.array(train_std)),
                    1 - (np.array(train_mean) + np.array(train_std)), 
                    alpha=0.1, color='blue', label='Training Variance')
    ax2.fill_between(train_sizes, 
                    1 - (np.array(val_mean) - np.array(val_std)),
                    1 - (np.array(val_mean) + np.array(val_std)), 
                    alpha=0.1, color='orange', label='Validation Variance')
    ax2.plot(train_sizes, [1 - score for score in train_mean], 'o-', color='blue', 
            label='Training Error', linewidth=2)
    ax2.plot(train_sizes, [1 - score for score in val_mean], 'o-', color='orange', 
            label='Validation Error', linewidth=2)
    ax2.set_title('Model Error vs Training Size', fontsize=14, pad=20)
    ax2.set_xlabel('Number of Training Examples', fontsize=12)
    ax2.set_ylabel('Error Rate', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim([0, 0.4])

    plt.tight_layout()
    return fig

# Display learning curves
if st.session_state.learning_curves_fig is None:
    with st.spinner('Generating learning curves...'):
        st.session_state.learning_curves_fig = plot_learning_curves()
        
st.pyplot(st.session_state.learning_curves_fig)

st.markdown("""
### Understanding the Learning Curves:
    
**Left Plot (Accuracy):**
- Higher values are better
- Shows how accurately the model predicts both training and validation data
- Converging lines indicate good model fit
    
**Right Plot (Error):**
- Lower values are better
- Shows the model's error rate on training and validation data
- Helps identify overfitting or underfitting
    
**What to Look For:**
- Small gap between training and validation curves
- Stable performance as training size increases
- Low variance (narrow shaded areas)
""")

# ROC Curve
st.subheader("📈 ROC Curve Analysis")

# Calculate ROC curve points and AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Create ROC curve plot
def plot_roc_curve():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_roc:.2f})', linewidth=2)
    ax.fill_between(fpr, tpr, alpha=0.2, color='blue')

    # Improve styling
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add more ticks
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Add minor gridlines
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax.minorticks_on()

    # Add annotation
    ax.annotate(f'AUC = {auc_roc:.2f}', 
               xy=(0.6, 0.2), 
               xytext=(0.6, 0.2),
               fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    plt.tight_layout()
    return fig

# Display ROC curve
if st.session_state.roc_curve_fig is None:
    with st.spinner('Generating ROC curve...'):
        st.session_state.roc_curve_fig = plot_roc_curve()
        
st.pyplot(st.session_state.roc_curve_fig)

# Add classification metrics table
st.subheader("📋 Model Performance Metrics")
if st.session_state.evaluation_metrics is not None:
    st.table(st.session_state.evaluation_metrics)

# Add final explanations
st.markdown("""
### Understanding the Metrics:

**Performance Metrics:**
- **Precision**: When the model predicts heart disease, how often is it correct?
- **Recall**: Out of all actual heart disease cases, how many did the model identify?
- **F1-Score**: Balance between precision and recall (1.0 is best, 0.0 is worst)
- **Support**: Number of samples in each class in the test dataset

**ROC Curve:**
- Shows the trade-off between true positive rate and false positive rate
- AUC (Area Under Curve) closer to 1.0 indicates better model performance
- Higher curve indicates better model discrimination ability
""")