import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Page configuration
st.set_page_config(page_title="IT Project Risk Prediction System", layout="wide")

# Title and description
st.title("IT Project Risk Prediction System")
st.markdown("""
This application predicts the risk level of IT projects based on project characteristics using a Logistic Regression model.
The model was trained on a synthetic dataset of 10,000 projects with features like budget, duration, team size, and methodology.
Prediction logic uses a multi-class classification (low, medium, high risk) based on standardized and one-hot encoded features.
""")

# Load model, preprocessor, and feature info
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/best_model_logistic_regression.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        if not hasattr(preprocessor, 'transformers_'):
            raise ValueError("Preprocessor is not fitted. Please run 'fit_preprocessor.py' to fit it.")
        with open('data/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        if 'final_features' not in feature_info or not isinstance(feature_info['final_features'], list):
            raise ValueError("data/feature_info.json is missing or has an invalid 'final_features' list.")
        return model, preprocessor, feature_info['final_features']
    except FileNotFoundError as e:
        st.error(f"Missing file: {str(e)}. Please ensure all required files are in the 'models' and 'data' directories.")
        return None, None, None
    except ValueError as e:
        st.error(f"Configuration error: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading assets: {str(e)}")
        return None, None, None

model, preprocessor, final_features = load_assets()

if model is None or preprocessor is None or final_features is None:
    st.error("""
    Failed to load required assets. Please ensure:
    1. 'data/X_train_processed.pkl' exists (8000 rows, 13+ features).
    2. 'fit_preprocessor.py' has been run to generate 'models/preprocessor.pkl'.
    3. 'models/best_model_logistic_regression.pkl' and 'data/feature_info.json' are present.
    4. Rerun the app after fixing the issues.
    """)
    st.stop()

# Input form with whole numbers
st.header("Enter Project Details")
with st.form("project_form"):
    col1, col2 = st.columns(2)

    with col1:
        planned_budget = st.number_input("Planned Budget (M$)", min_value=1, max_value=5, value=1, step=1) * 1000000
        planned_duration = st.number_input("Planned Duration (weeks)", min_value=4, max_value=6, value=4, step=2)
        team_size = st.number_input("Team Size", min_value=5, max_value=50, value=10, step=1)
        technical_complexity = st.number_input("Technical Complexity (1-10)", min_value=1, max_value=10, value=5, step=1)
        stakeholder_count = st.number_input("Stakeholder Count", min_value=1, max_value=20, value=5, step=1)

    with col2:
        methodology = st.selectbox("Methodology", options=[
            'Agile', 'Waterfall', 'Spiral', 'Iterative', 'V-Model', 'Prototyping'
        ])
        project_type = st.selectbox("Project Type", options=[
            'Web Development', 'AI/ML', 'Cloud Migration', 'Mobile App', 'Data Analytics',
            'Cybersecurity', 'DevOps', 'Blockchain', 'IoT', 'Enterprise Software'
        ])

    submit_button = st.form_submit_button("Predict Risk")

# Process inputs and predict
if submit_button:
    st.header("Prediction Results")
    
    # Create input dictionary with unprefixed feature names
    input_data = {
        'planned_budget': planned_budget,
        'planned_duration': planned_duration,
        'team_size': team_size,
        'technical_complexity': technical_complexity,
        'stakeholder_count': stakeholder_count,
        'budget_per_person': planned_budget / team_size,
        'complexity_index': technical_complexity * 0.4 + stakeholder_count * 0.2,
        'team_efficiency': technical_complexity / team_size,
        'stakeholder_density': stakeholder_count / team_size,
        'timeline_pressure': technical_complexity / planned_duration,
        'budget_complexity_ratio': planned_budget / (technical_complexity * 1000),
        'methodology': methodology,
        'project_type': project_type
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Debug: Print columns to verify
    st.write("Input DataFrame columns before transformation:", input_df.columns.tolist())
    
    try:
        # Preprocess input
        input_processed = preprocessor.transform(input_df)
        input_processed_df = pd.DataFrame(input_processed, columns=preprocessor.get_feature_names_out())
        st.write("Processed DataFrame columns:", input_processed_df.columns.tolist())

        # Predict
        risk_proba = model.predict_proba(input_processed_df)
        risk_class_map = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        risk_class_idx = np.argmax(risk_proba, axis=1)[0]
        risk_class = risk_class_map[risk_class_idx]
        risk_proba_display = f"{risk_proba[0, risk_class_idx]:.2%}"

        # Display results
        st.subheader("Risk Assessment")
        st.write(f"**Risk Probability**: {risk_proba_display}")
        st.write(f"**Risk Classification**: {risk_class}")
        
        # Model performance note
        st.markdown("""
        **Note**: The current model (Logistic Regression) achieved a test ROC AUC of ~0.98 on synthetic data.
        This high performance may not generalize to real-world data. Consider retraining with real data for accuracy.
        """)

        # Suggestions based on risk level
        st.subheader("Project Suggestions")
        if risk_class == 'Low Risk':
            st.write("Great job! Maintain current practices and consider scaling the project if needed.")
        elif risk_class == 'Medium Risk':
            st.write("Proceed with caution. Enhance risk monitoring and consider iterative reviews to mitigate issues.")
        elif risk_class == 'High Risk':
            st.write("High risk detected. Reassess scope, reduce complexity, or switch to an Agile methodology with frequent feedback loops.")

        # Save prediction result
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'risk_probability': risk_proba_display,
            'risk_classification': risk_class,
            'planned_budget': planned_budget // 1000000,  # Convert back to M$
            'planned_duration': planned_duration,
            'team_size': team_size,
            'technical_complexity': technical_complexity,
            'stakeholder_count': stakeholder_count,
            'methodology': methodology,
            'project_type': project_type
        }
        result_df = pd.DataFrame([result])
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download Prediction Result",
            data=csv_buffer.getvalue(),
            file_name="project_risk_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing prediction: {str(e)}. Please ensure input features match the training data. Check 'data/feature_info.json' for the exact feature list.")

# Footer
st.markdown("---")
st.markdown("Developed by emma")