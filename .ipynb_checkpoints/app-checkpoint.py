import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load("models/rf_model.pkl")

# Load the external CSS file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <header>
        <h1>IT Projects Risk Management</h1>
    </header>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.markdown(
    """
    <div class="sidebar">
        <h2>Welcome</h2>
        <p>This dashboard helps project managers predict IT project risks and optimize resources. Input project details to receive AI-driven insights and actionable recommendations for cost, team size, and duration adjustments.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main Content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Title and Header
st.markdown("<h1 class='text-center'>Project Risk & Resource Optimization</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='text-center'>Use AI to predict risks and optimize resources for your IT projects</h3>", unsafe_allow_html=True)

# Input Form
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h2>Enter Project Details</h2>", unsafe_allow_html=True)
with st.form("project_form"):
    project_name = st.text_input("Project Name", value="My IT Project")
    col1, col2, col3 = st.columns(3)
    with col1:
        team_size = st.number_input("Team Size", min_value=1, max_value=55, value=10)
    with col2:
        planned_duration = st.number_input("Planned Duration (Days)", min_value=0, max_value=975, value=180)
    with col3:
        planned_budget = st.number_input("Planned Budget (USD)", min_value=0, max_value=15000000, value=1000000, format="%d")
    
    col4, col5 = st.columns(2)
    with col4:
        complexity = st.slider("Project Complexity", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    with col5:
        methodology = st.selectbox("Methodology", ["Agile", "Waterfall", "Hybrid", "Scrum", "Kanban", "DevOps", "Lean"])
    
    industry = st.selectbox("Industry", [
        "Finance", "Healthcare", "Technology", "Retail", "Manufacturing",
        "Government", "Education", "Telecommunications", "Insurance", "Energy",
        "Transportation", "Media", "Pharmaceutical", "Consulting", "Entertainment"
    ])
    
    st.markdown("<h2>Risk Factors (Scale: 1 to 10)</h2>", unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    with col6:
        technical_risk = st.slider("Technical Risk", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
        communication_risk = st.slider("Communication Risk", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    with col7:
        requirement_risk = st.slider("Requirement Risk", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
        vendor_risk = st.slider("Vendor Risk", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    with col8:
        methodology_risk = st.slider("Methodology Risk", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
        client_experience = st.slider("Client Experience", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    
    vendor_dependency = st.slider("Vendor Dependency (Scale: 0 to 10)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    stakeholder_count = st.number_input("Stakeholder Count", min_value=1, max_value=21, value=5)

    # Form submission
    submitted = st.form_submit_button("Analyze Project")
st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    # Prepare input data for prediction
    input_data = {
        'team_size': team_size,
        'planned_duration': planned_duration,
        'planned_budget': planned_budget,
        'start_year': datetime.now().year,  # Current year as placeholder
        'industry': industry,
        'methodology': methodology,
        'complexity': complexity,
        'client_experience': client_experience,
        'vendor_dependency': vendor_dependency,
        'stakeholder_count': stakeholder_count,
        'technical_risk': technical_risk,
        'communication_risk': communication_risk,
        'requirement_risk': requirement_risk,
        'vendor_risk': vendor_risk,
        'methodology_risk': methodology_risk,
        # Placeholder values for features not directly input by user
        'budget_overrun_pct': 0.0,
        'actual_budget': planned_budget,
        'schedule_delay_pct': 0.0,
        'actual_duration': planned_duration,
        'scope_delivered_pct': 100.0,
        'quality_score': 8.0,
        'customer_satisfaction': 8.0,
        'success_rating': 8.0,
        'planned_duration_months': planned_duration / 30.44,
        'composite_risk_score': (technical_risk + communication_risk + requirement_risk + vendor_risk + methodology_risk) / 5,
        'budget_per_team_member': planned_budget / team_size,
        'duration_complexity_factor': planned_duration * complexity / 100,
        'risk_assessment_variance': np.var([technical_risk, communication_risk, requirement_risk, vendor_risk, methodology_risk]),
        'experience_complexity_gap': complexity - client_experience,
        'budget_category': 'Medium',  # Placeholder, will be encoded
        'duration_category': 'Medium',
        'team_size_category': 'Medium'
    }

    input_df = pd.DataFrame([input_data])

    # Predict risk
    risk_prediction = model.predict(input_df)[0]
    risk_prob = model.predict_proba(input_df)[0][1]

    # Display Results
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Risk Prediction</h2>", unsafe_allow_html=True)
    if risk_prob >= 0.7:
        risk_class = "risk-high"
        risk_label = "⚠️ High Risk Detected"
    elif risk_prob >= 0.4:
        risk_class = "risk-medium"
        risk_label = "⚠️ Medium Risk Detected"
    else:
        risk_class = "risk-low"
        risk_label = "✅ Low Risk"
    st.markdown(f'<div class="{risk_class}">{risk_label} (Probability: {risk_prob:.2f})</div>', unsafe_allow_html=True)

    # Visualizations
    st.markdown("<h2>Risk Factor Analysis</h2>", unsafe_allow_html=True)
    risk_factors = {
        'Technical Risk': technical_risk,
        'Communication Risk': communication_risk,
        'Requirement Risk': requirement_risk,
        'Vendor Risk': vendor_risk,
        'Methodology Risk': methodology_risk
    }
    risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Factor', 'Score'])
    fig = px.bar(risk_df, x='Score', y='Risk Factor', orientation='h',
                 title='Risk Factor Scores',
                 color='Score', color_continuous_scale='Reds',
                 labels={'Score': 'Risk Score (1-10)', 'Risk Factor': ''},
                 height=400)
    fig.update_layout(
        title={'text': 'Risk Factor Scores', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
        margin=dict(t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Open Sans', size=14, color='#2c3e50')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Budget Allocation Pie Chart
    st.markdown("<h2>Budget Allocation Overview</h2>", unsafe_allow_html=True)
    team_cost = team_size * planned_duration * 500  # Assume $500 per person per day
    other_cost = planned_budget - team_cost
    budget_data = pd.DataFrame({
        'Category': ['Team Costs', 'Other Costs'],
        'Amount': [team_cost, max(other_cost, 0)]
    })
    fig2 = px.pie(budget_data, values='Amount', names='Category',
                  title='Budget Allocation',
                  color_discrete_sequence=['#3498db', '#95a5a6'],
                  height=400)
    fig2.update_layout(
        title={'text': 'Budget Allocation', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
        margin=dict(t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Open Sans', size=14, color='#2c3e50')
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Suggestions
    st.markdown("<h2>Actionable Recommendations</h2>", unsafe_allow_html=True)
    suggestions = []
    composite_risk = input_data['composite_risk_score']

    # Team Size Suggestions
    if team_size < 5 and composite_risk > 5:
        suggestions.append("Increase team size by 2 to handle high complexity and reduce schedule delay risk.")
    elif team_size > 20 and composite_risk < 3:
        suggestions.append("Consider reducing team size by 2 to optimize costs without increasing risk.")

    # Budget Suggestions
    budget_per_person = planned_budget / team_size
    if budget_per_person < 50000 and risk_prediction == 1:
        suggestions.append("Increase budget by 10% to mitigate risks associated with underfunding.")
    elif budget_per_person > 100000 and risk_prediction == 0:
        suggestions.append("Decrease budget by 5% to optimize costs, as risk is low.")

    # Duration Suggestions
    if planned_duration < 100 and composite_risk > 5:
        suggestions.append("Extend duration by 10 days to reduce pressure on the team and lower technical risk.")
    elif planned_duration > 500 and composite_risk < 3:
        suggestions.append("Shorten duration by 10 days to save costs, as risk is manageable.")

    if not suggestions:
        suggestions.append("No immediate adjustments needed. Continue monitoring project progress.")
    for suggestion in suggestions:
        st.markdown(f'<div class="suggestion">{suggestion}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Monitoring and Maintenance Recommendations
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h2>Model Monitoring & Maintenance</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <ul>
        <li><strong>Regular Updates:</strong> Retrain the model with real project data to improve accuracy over time.</li>
        <li><strong>Feedback Loop:</strong> Gather feedback from project managers to refine recommendations.</li>
        <li><strong>Advanced Techniques:</strong> Consider using SMOTE for class imbalance or Gradient Boosting for enhanced performance.</li>
    </ul>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)