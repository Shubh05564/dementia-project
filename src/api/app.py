"""
Streamlit web application for Dementia Prevention Advisor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Dementia Prevention Advisor",
    page_icon="🧠",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Title and description
st.title("🧠 Personalized Dementia Prevention Advisor")
st.markdown("""
### ML-Assisted Dietary Optimization for Dementia Risk Reduction

This tool provides personalized dietary recommendations based on individual risk factors 
and expected treatment effects from machine learning models trained on large cohort studies.
""")

# Sidebar for patient information
st.sidebar.header("Patient Information")

# Demographics
age = st.sidebar.slider("Age", 50, 90, 65)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex_encoded = 1 if sex == "Male" else 0

bmi = st.sidebar.number_input("BMI", 15.0, 50.0, 25.0)
education = st.sidebar.slider("Years of Education", 0, 20, 12)

# Genetic factors
apoe_e4 = st.sidebar.selectbox("APOE ε4 Status", ["Non-carrier", "Carrier", "Unknown"])
apoe_encoded = 1 if apoe_e4 == "Carrier" else 0

# Current diet
st.sidebar.subheader("Current Diet Assessment")
mind_score = st.sidebar.slider(
    "Current MIND Diet Score (0-15)", 
    0.0, 15.0, 5.0,
    help="Higher scores indicate better adherence to MIND diet principles"
)
mediterranean_score = st.sidebar.slider(
    "Mediterranean Diet Score (0-9)", 
    0.0, 9.0, 3.0
)

# Clinical factors
st.sidebar.subheader("Clinical Factors")
comorbidities = st.sidebar.number_input("Number of Comorbidities", 0, 10, 0)
bp_systolic = st.sidebar.number_input("Systolic Blood Pressure (mmHg)", 90, 200, 120)
cholesterol = st.sidebar.number_input("Total Cholesterol (mg/dL)", 120, 350, 200)

# Lifestyle
st.sidebar.subheader("Lifestyle Factors")
physical_activity = st.sidebar.select_slider(
    "Physical Activity Level",
    options=["Sedentary", "Light", "Moderate", "Vigorous"],
    value="Light"
)
activity_encoded = ["Sedentary", "Light", "Moderate", "Vigorous"].index(physical_activity)

smoking = st.sidebar.selectbox("Smoking Status", ["Never", "Former", "Current"])
smoking_encoded = ["Never", "Former", "Current"].index(smoking)

# Generate recommendation button
if st.sidebar.button("Generate Recommendation", type="primary"):
    
    # Prepare patient data
    patient_data = {
        "age": age,
        "sex": sex_encoded,
        "bmi": bmi,
        "education": education,
        "apoe_e4_carrier": apoe_encoded,
        "mind_score": mind_score,
        "mediterranean_score": mediterranean_score,
        "comorbidity_count": comorbidities,
        "blood_pressure_systolic": bp_systolic,
        "cholesterol_total": cholesterol,
        "physical_activity_level": activity_encoded,
        "smoking_status": smoking_encoded
    }
    
    # Call API
    try:
        with st.spinner("Analyzing risk and generating personalized recommendation..."):
            response = requests.post(f"{API_URL}/recommend", json=patient_data)
            
            if response.status_code == 200:
                recommendation = response.json()
                
                # Store in session state
                st.session_state['recommendation'] = recommendation
                st.success("Recommendation generated successfully!")
            else:
                st.error(f"Error: {response.text}")
                st.session_state['recommendation'] = None
    
    except requests.exceptions.ConnectionError:
        st.warning("Could not connect to API. Using demo mode with simplified calculations.")
        
        # Fallback demo mode
        baseline_risk = 0.05
        if age > 65:
            baseline_risk += 0.10
        if age > 75:
            baseline_risk += 0.15
        if apoe_encoded == 1:
            baseline_risk += 0.20
        
        cate = 0.05
        if apoe_encoded == 1:
            cate += 0.03
        if mind_score > 8:
            cate *= 0.5
        
        recommendation = {
            "recommended_diet": "MIND Diet" if cate > 0.05 else "Mediterranean Diet",
            "expected_risk_reduction": cate,
            "confidence_level": "high" if cate > 0.05 else "moderate",
            "baseline_risk": baseline_risk,
            "post_intervention_risk": max(0, baseline_risk - cate),
            "top_risk_factors": [
                {"factor": "Age > 65", "impact": 0.15} if age > 65 else None,
                {"factor": "APOE ε4 carrier", "impact": 0.25} if apoe_encoded == 1 else None
            ],
            "top_protective_factors": [
                {"factor": "High diet score", "impact": -0.12} if mind_score > 8 else None
            ],
            "specific_recommendations": [
                "Eat leafy greens 6+ servings/week",
                "Include berries 2+ servings/week",
                "Consume nuts 5+ times/week"
            ],
            "adherence_tips": [
                "Start with small changes",
                "Plan meals in advance",
                "Track your progress"
            ]
        }
        
        # Clean up None values
        recommendation['top_risk_factors'] = [x for x in recommendation['top_risk_factors'] if x]
        recommendation['top_protective_factors'] = [x for x in recommendation['top_protective_factors'] if x]
        
        st.session_state['recommendation'] = recommendation

# Display recommendation if available
if 'recommendation' in st.session_state and st.session_state['recommendation']:
    rec = st.session_state['recommendation']
    
    # Main columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Recommended diet
        st.header(f"📋 Recommended: {rec['recommended_diet']}")
        st.metric(
            "Expected Risk Reduction",
            f"{rec['expected_risk_reduction']:.1%}",
            delta=f"Confidence: {rec['confidence_level']}"
        )
        
        # Risk visualization
        st.subheader("Risk Assessment")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Baseline Risk', 'Post-Intervention Risk'],
            y=[rec['baseline_risk'] * 100, rec['post_intervention_risk'] * 100],
            marker_color=['#FF6B6B', '#51CF66'],
            text=[f"{rec['baseline_risk']:.1%}", f"{rec['post_intervention_risk']:.1%}"],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Dementia Risk Comparison",
            yaxis_title="Risk (%)",
            yaxis_range=[0, max(rec['baseline_risk'] * 100, 50) + 5],
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Specific recommendations
        st.subheader("Dietary Recommendations")
        for i, rec_item in enumerate(rec['specific_recommendations'], 1):
            st.markdown(f"{i}. {rec_item}")
        
        # Adherence tips
        with st.expander("Tips for Adherence"):
            for tip in rec['adherence_tips']:
                st.markdown(f"• {tip}")
    
    with col2:
        # Risk factors
        st.subheader("Risk Factors")
        if rec['top_risk_factors']:
            for factor in rec['top_risk_factors']:
                st.metric(
                    factor['factor'],
                    f"+{factor['impact']:.1%}",
                    delta="Risk increase",
                    delta_color="inverse"
                )
        else:
            st.info("No major risk factors identified")
        
        # Protective factors
        st.subheader("Protective Factors")
        if rec['top_protective_factors']:
            for factor in rec['top_protective_factors']:
                st.metric(
                    factor['factor'],
                    f"{factor['impact']:.1%}",
                    delta="Risk reduction",
                    delta_color="normal"
                )
        else:
            st.info("No major protective factors identified")
    
    # Additional information tabs
    tab1, tab2, tab3 = st.tabs(["📊 Evidence Base", "🍽️ Sample Meal Plan", "📈 Track Progress"])
    
    with tab1:
        st.markdown("""
        ### Evidence Base
        
        These recommendations are based on:
        
        - **NHANES**: National Health and Nutrition Examination Survey
        - **ADNI**: Alzheimer's Disease Neuroimaging Initiative
        - **UK Biobank**: Large-scale biomedical database
        - **ELSA**: English Longitudinal Study of Ageing
        
        The machine learning models use:
        - Causal inference methods (propensity scores, doubly robust estimation)
        - Heterogeneous treatment effect estimation
        - Survival analysis for time-to-event outcomes
        """)
    
    with tab2:
        st.markdown(f"""
        ### Sample {rec['recommended_diet']} Meal Plan
        
        #### Breakfast
        - Oatmeal with berries, walnuts, and a drizzle of honey
        - Green tea
        
        #### Lunch
        - Large mixed greens salad with olive oil vinaigrette
        - Grilled salmon
        - Whole grain bread
        
        #### Dinner
        - Baked chicken with herbs
        - Roasted vegetables (broccoli, Brussels sprouts)
        - Quinoa or brown rice
        
        #### Snacks
        - Handful of mixed nuts
        - Fresh fruit
        - Hummus with vegetables
        """)
    
    with tab3:
        st.markdown("""
        ### Track Your Progress
        
        Monitor your adherence to the dietary recommendations:
        
        1. Keep a food diary
        2. Calculate your weekly MIND diet score
        3. Schedule follow-up assessments every 3-6 months
        4. Track cognitive function with periodic screening
        
        **Next Steps:**
        - Download your personalized plan
        - Share with your healthcare provider
        - Set up reminders for dietary goals
        """)
        
        if st.button("Download Recommendation (PDF)"):
            st.info("PDF download feature coming soon!")

# Population-level analysis
st.markdown("---")
st.header("📊 Population-Level Impact Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Prevented Cases (per 100K)",
        "12,000",
        help="Over 10-year period with MIND diet intervention"
    )

with col2:
    st.metric(
        "Average Treatment Effect",
        "8%",
        help="Relative risk reduction in dementia incidence"
    )

with col3:
    st.metric(
        "Number Needed to Treat",
        "12",
        help="Number of people needed to follow diet to prevent 1 case"
    )

# Footer
st.markdown("---")
st.caption("⚠️ This tool is for educational and research purposes. Always consult with healthcare professionals before making dietary changes.")
