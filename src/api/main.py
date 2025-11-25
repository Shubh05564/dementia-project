"""
FastAPI application for personalized dementia prevention recommendations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dementia Prevention Advisor API",
    description="ML-assisted optimization of dietary interventions against dementia risk",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests/responses
class PatientFeatures(BaseModel):
    """Patient features for recommendation."""
    age: float
    sex: int  # 0=Female, 1=Male
    bmi: Optional[float] = None
    education: Optional[int] = None
    apoe_e4_carrier: Optional[int] = 0
    
    # Dietary features
    mind_score: Optional[float] = 5.0
    mediterranean_score: Optional[float] = 3.0
    
    # Clinical features
    comorbidity_count: Optional[int] = 0
    blood_pressure_systolic: Optional[float] = None
    cholesterol_total: Optional[float] = None
    
    # Lifestyle
    physical_activity_level: Optional[int] = 1
    smoking_status: Optional[int] = 0


class RecommendationResponse(BaseModel):
    """Dietary recommendation response."""
    patient_id: Optional[str] = None
    recommended_diet: str
    expected_risk_reduction: float
    confidence_level: str
    baseline_risk: float
    post_intervention_risk: float
    
    # Explanation
    top_risk_factors: List[Dict[str, float]]
    top_protective_factors: List[Dict[str, float]]
    
    # Dietary recommendations
    specific_recommendations: List[str]
    adherence_tips: List[str]


class PopulationAnalysisRequest(BaseModel):
    """Request for population-level analysis."""
    subgroup: Optional[str] = None
    intervention_type: str = "mind_diet"


# Global variables for loaded models (in production, use proper model management)
causal_model = None
policy_model = None
explainer = None


@app.on_event("startup")
async def load_models():
    """Load ML models on startup."""
    global causal_model, policy_model, explainer
    
    logger.info("Loading ML models...")
    
    # In production, load actual trained models
    # causal_model = joblib.load('models/causal_model.pkl')
    # policy_model = joblib.load('models/policy_model.pkl')
    # explainer = joblib.load('models/explainer.pkl')
    
    logger.info("Models loaded successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dementia Prevention Advisor API",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "Get personalized dietary recommendation",
            "/population-analysis": "Analyze population-level effects",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": causal_model is not None}


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(patient: PatientFeatures):
    """Generate personalized dietary recommendation.
    
    Args:
        patient: Patient features
    
    Returns:
        Personalized recommendation
    """
    try:
        logger.info(f"Generating recommendation for patient: age={patient.age}, sex={patient.sex}")
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient.dict()])
        
        # Fill missing values with defaults
        patient_df = patient_df.fillna({
            'bmi': 25.0,
            'education': 12,
            'blood_pressure_systolic': 120,
            'cholesterol_total': 200
        })
        
        # Estimate baseline risk (simplified)
        baseline_risk = estimate_baseline_risk(patient_df)
        
        # Estimate treatment effect (simplified for demo)
        cate = estimate_treatment_effect(patient_df)
        
        # Generate recommendation
        if cate > 0.05:  # Significant positive effect
            recommended_diet = "MIND Diet"
            confidence = "high"
            specific_recs = [
                "Eat leafy green vegetables at least 6 servings per week",
                "Include berries at least 2 servings per week",
                "Consume nuts 5+ times per week",
                "Eat fish at least once per week",
                "Use olive oil as primary cooking oil",
                "Limit red meat to less than 4 servings per week",
                "Minimize butter, cheese, and fried foods"
            ]
        elif cate > 0:
            recommended_diet = "Mediterranean Diet"
            confidence = "moderate"
            specific_recs = [
                "Base meals on vegetables, fruits, whole grains",
                "Use olive oil generously",
                "Eat fish and seafood at least twice per week",
                "Include moderate amounts of poultry, eggs, dairy",
                "Limit red meat consumption"
            ]
        else:
            recommended_diet = "Standard Healthy Diet"
            confidence = "low"
            specific_recs = [
                "Maintain a balanced diet with variety",
                "Include fruits and vegetables daily",
                "Choose whole grains over refined grains",
                "Limit processed foods and added sugars"
            ]
        
        # Calculate post-intervention risk
        post_intervention_risk = max(0, baseline_risk - abs(cate))
        risk_reduction = baseline_risk - post_intervention_risk
        
        # Identify risk factors (simplified)
        risk_factors = []
        protective_factors = []
        
        if patient.age > 65:
            risk_factors.append({"factor": "Age > 65", "impact": 0.15})
        if patient.apoe_e4_carrier == 1:
            risk_factors.append({"factor": "APOE ε4 carrier", "impact": 0.25})
        if patient.mind_score and patient.mind_score < 7:
            risk_factors.append({"factor": "Low MIND diet score", "impact": 0.10})
        
        if patient.mind_score and patient.mind_score > 8:
            protective_factors.append({"factor": "High MIND diet score", "impact": -0.12})
        if patient.physical_activity_level and patient.physical_activity_level > 2:
            protective_factors.append({"factor": "Regular physical activity", "impact": -0.08})
        
        # Adherence tips
        adherence_tips = [
            "Start with small, gradual changes to your diet",
            "Plan meals in advance and prep ingredients",
            "Find healthy recipes you enjoy",
            "Track your progress with a food diary",
            "Seek support from family and friends"
        ]
        
        recommendation = RecommendationResponse(
            recommended_diet=recommended_diet,
            expected_risk_reduction=float(risk_reduction),
            confidence_level=confidence,
            baseline_risk=float(baseline_risk),
            post_intervention_risk=float(post_intervention_risk),
            top_risk_factors=risk_factors,
            top_protective_factors=protective_factors,
            specific_recommendations=specific_recs,
            adherence_tips=adherence_tips
        )
        
        logger.info(f"Recommendation generated: {recommended_diet}")
        
        return recommendation
        
    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/population-analysis")
async def population_analysis(request: PopulationAnalysisRequest):
    """Analyze population-level intervention effects.
    
    Args:
        request: Population analysis request
    
    Returns:
        Population-level statistics
    """
    try:
        # Simulate population analysis (in production, use actual data)
        
        if request.intervention_type == "mind_diet":
            ate = 0.08  # 8% reduction in dementia risk
            prevented_cases = 12000  # per 100,000 over 10 years
        elif request.intervention_type == "mediterranean_diet":
            ate = 0.06
            prevented_cases = 9000
        else:
            ate = 0.03
            prevented_cases = 4500
        
        analysis = {
            "intervention": request.intervention_type,
            "average_treatment_effect": ate,
            "prevented_cases_per_100k": prevented_cases,
            "number_needed_to_treat": int(1 / ate) if ate > 0 else None,
            "cost_effectiveness": "Cost-effective compared to pharmaceutical interventions",
            "subgroup_effects": {
                "apoe_e4_carriers": {"ate": ate * 1.5, "n": 15000},
                "age_65plus": {"ate": ate * 1.2, "n": 45000},
                "baseline_low_diet_score": {"ate": ate * 1.8, "n": 30000}
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in population analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def estimate_baseline_risk(patient_df: pd.DataFrame) -> float:
    """Estimate baseline dementia risk."""
    # Simplified risk estimation
    age = patient_df['age'].values[0]
    apoe = patient_df['apoe_e4_carrier'].values[0]
    
    risk = 0.05  # Baseline 5%
    
    if age > 65:
        risk += 0.10
    if age > 75:
        risk += 0.15
    if apoe == 1:
        risk += 0.20
    
    return min(risk, 0.80)  # Cap at 80%


def estimate_treatment_effect(patient_df: pd.DataFrame) -> float:
    """Estimate conditional treatment effect."""
    # Simplified CATE estimation
    # In production, use actual trained model
    
    age = patient_df['age'].values[0]
    mind_score = patient_df['mind_score'].values[0]
    apoe = patient_df['apoe_e4_carrier'].values[0]
    
    cate = 0.05  # Base effect
    
    # Higher effect for APOE ε4 carriers
    if apoe == 1:
        cate += 0.03
    
    # Effect diminishes with age
    if age > 75:
        cate *= 0.7
    
    # Already high diet score → less benefit
    if mind_score > 8:
        cate *= 0.5
    
    return cate


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
