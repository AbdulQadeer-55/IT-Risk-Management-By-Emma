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

# Page configuration (must be the first Streamlit command)
st.set_page_config(page_title="IT Project Risk Prediction System", layout="wide")

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Header styling */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #34495e;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    
    /* Input fields */
    .stNumberInput, .stSelectbox {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #dfe6e9;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
    }
    .stDownloadButton > button:hover {
        background-color: #27ae60;
    }
    
    /* Error messages */
    .stAlert {
        background-color: #ffebee;
        color: #c0392b;
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Expander */
    .stExpander {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #dfe6e9;
    }
    
    /* Table styling */
    .stTable {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* General text */
    body, p, div {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
with st.container():
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

# Pre-dashboard for input preview
with st.container():
    st.header("Pre-Dashboard: Input Preview")
    with st.expander("View Input Preview", expanded=True):
        preview_data = {
            'Planned Budget': 'Not set',
            'Planned Duration': 'Not set',
            'Team Size': 'Not set',
            'Technical Complexity': 'Not set',
            'Stakeholder Count': 'Not set',
            'Methodology': 'Not set',
            'Project Type': 'Not set'
        }
        preview_df = pd.DataFrame([preview_data])
        st.table(preview_df)

# Input form with whole numbers and tooltips
with st.container():
    st.header("Enter Project Details")
    with st.form("project_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Budget range selection
            budget_ranges = [
                ('0.5M - 1M', 750000),
                ('1M - 2M', 1500000),
                ('2M - 3M', 2500000),
                ('3M - 4M', 3500000),
                ('4M - 5M', 4500000)
            ]
            budget_label = st.selectbox(
                "Planned Budget Range",
                options=[r[0] for r in budget_ranges],
                help="Select the budget range in millions of dollars. This is the total planned budget for the project."
            )
            planned_budget = next(r[1] for r in budget_ranges if r[0] == budget_label)

            planned_duration = st.number_input(
                "Planned Duration (weeks)",
                min_value=4, max_value=6, value=4, step=2,
                help="Enter the planned project duration in weeks (4-6 weeks, in steps of 2)."
            )
            team_size = st.number_input(
                "Team Size",
                min_value=5, max_value=50, value=10, step=1,
                help="Enter the number of team members (5-50 people)."
            )
            technical_complexity = st.number_input(
                "Technical Complexity (1-10)",
                min_value=1, max_value=10, value=5, step=1,
                help="Rate the technical complexity of the project on a scale of 1 (simple) to 10 (highly complex)."
            )
            stakeholder_count = st.number_input(
                "Stakeholder Count",
                min_value=1, max_value=20, value=5, step=1,
                help="Enter the number of stakeholders involved (1-20)."
            )

        with col2:
            methodology = st.selectbox(
                "Methodology",
                options=['Agile', 'Waterfall', 'Spiral', 'Iterative', 'V-Model', 'Prototyping'],
                help="Select the project management methodology."
            )
            project_type = st.selectbox(
                "Project Type",
                options=[
                    'Web Development', 'AI/ML', 'Cloud Migration', 'Mobile App', 'Data Analytics',
                    'Cybersecurity', 'DevOps', 'Blockchain', 'IoT', 'Enterprise Software'
                ],
                help="Select the type of IT project."
            )

        submit_button = st.form_submit_button("Predict Risk")

        # Update preview data with formatted budget
        preview_data.update({
            'Planned Budget': budget_label,
            'Planned Duration': f"{planned_duration} weeks",
            'Team Size': team_size,
            'Technical Complexity': technical_complexity,
            'Stakeholder Count': stakeholder_count,
            'Methodology': methodology,
            'Project Type': project_type
        })
        preview_df = pd.DataFrame([preview_data])
        st.session_state['preview_df'] = preview_df

# Display updated preview
if 'preview_df' in st.session_state:
    with st.container():
        with st.expander("View Input Preview", expanded=True):
            st.table(st.session_state['preview_df'])

# Process inputs and predict
if submit_button:
    # Input validation
    errors = []
    if planned_budget < 500000 or planned_budget > 5000000:
        errors.append("Planned Budget must be within $0.5M - $5M.")
    if planned_duration not in range(4, 7, 2):
        errors.append("Planned Duration must be 4 or 6 weeks.")
    if team_size < 5 or team_size > 50:
        errors.append("Team Size must be between 5 and 50.")
    if technical_complexity < 1 or technical_complexity > 10:
        errors.append("Technical Complexity must be between 1 and 10.")
    if stakeholder_count < 1 or stakeholder_count > 20:
        errors.append("Stakeholder Count must be between 1 and 20.")

    if errors:
        st.error("\n".join(errors))
    else:
        with st.container():
            st.header("Prediction Results")
            
            # Create input dictionary
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
            
            try:
                # Preprocess input
                input_processed = preprocessor.transform(input_df)
                input_processed_df = pd.DataFrame(input_processed, columns=preprocessor.get_feature_names_out())

                # Predict
                risk_proba = model.predict_proba(input_processed_df)
                risk_class_map = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
                risk_class_idx = np.argmax(risk_proba, axis=1)[0]
                risk_class = risk_class_map[risk_class_idx]
                risk_proba_display = f"{risk_proba[0, risk_class_idx]:.2%}"

                # Display results
                st.subheader("Risk Assessment")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Risk Probability**: {risk_proba_display}")
                with col2:
                    st.markdown(f"**Risk Classification**: {risk_class}")
                
                # Model performance note
                st.markdown("""
                **Note**: The current model (Logistic Regression) achieved a test ROC AUC of ~0.98 on synthetic data.
                This high performance may not generalize to real-world data. Consider retraining with real data for accuracy.
                """)

                # Visual report
                st.subheader("Visual Report")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Pie chart for risk probabilities
                labels = ['Low Risk', 'Medium Risk', 'High Risk']
                sizes = risk_proba[0]
                colors = ['#66b3ff', '#ffcc99', '#ff9999']
                explode = (0.1 if risk_class == 'Low Risk' else 0, 
                          0.1 if risk_class == 'Medium Risk' else 0, 
                          0.1 if risk_class == 'High Risk' else 0)
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode)
                ax1.axis('equal')
                ax1.set_title('Risk Probability Distribution', fontsize=14, pad=10)
                
                # Bar plot for key features
                key_features = {
                    'Budget (M$)': planned_budget / 1000000,
                    'Duration (weeks)': planned_duration,
                    'Team Size': team_size,
                    'Complexity': technical_complexity,
                    'Stakeholders': stakeholder_count
                }
                sns.barplot(x=list(key_features.values()), y=list(key_features.keys()), ax=ax2, palette='Blues_d')
                ax2.set_title('Key Project Features', fontsize=14, pad=10)
                ax2.set_xlabel('Value', fontsize=12)
                ax2.set_ylabel('Feature', fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig)

                # Suggestions based on risk level
                st.subheader("Project Suggestions")
                if risk_class == 'Low Risk':
                    st.success("Great job! Maintain current practices and consider scaling the project if needed.")
                elif risk_class == 'Medium Risk':
                    st.warning("Proceed with caution. Enhance risk monitoring and consider iterative reviews to mitigate issues.")
                elif risk_class == 'High Risk':
                    st.error("High risk detected. Reassess scope, reduce complexity, or switch to an Agile methodology with frequent feedback loops.")

                # Save prediction result
                result = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'risk_probability': risk_proba_display,
                    'risk_classification': risk_class,
                    'planned_budget': budget_label,
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
with st.container():
    st.markdown("---")
    st.markdown("**Developed by Emma** | Powered by Streamlit", unsafe_allow_html=True)