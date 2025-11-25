# Machine Learning-Assisted Optimization of Dietary Intervention Against Dementia Risk

## Overview
This project builds an interpretable ML pipeline that produces personalized dietary intervention rules to maximize expected reduction in dementia risk while accounting for adherence, safety, and resource constraints.

## Domain
- Precision Nutrition & Public Health
- Clinical Epidemiology & Dementia Prevention
- Causal Machine Learning / Decision Support Systems

## Key Features
1. **Personalized Treatment Effect Estimation**: Individual-level causal effect estimation
2. **Causal Inference Framework**: Propensity scores, IPW, doubly robust methods
3. **Policy Optimization**: Constrained optimization for feasible dietary recommendations
4. **Interpretability**: SHAP, PDP, subgroup analyses
5. **Deployment Ready**: FastAPI backend + Streamlit frontend

## Project Structure
```
dementia_nutrition_ml/
├── src/
│   ├── data/              # Data preprocessing & harmonization
│   ├── models/            # ML models for HTE estimation
│   ├── causal/            # Causal inference methods
│   ├── policy/            # Policy learning & optimization
│   ├── explainability/    # Interpretability tools
│   └── api/               # Deployment API
├── notebooks/             # Exploratory analysis
├── configs/               # Configuration files
├── data/                  # Data storage
├── tests/                 # Unit tests
└── requirements.txt       # Dependencies
```

## Installation
```bash
pip install -r requirements.txt
```

## Data Sources
- **NHANES**: https://wwwn.cdc.gov/nchs/nhanes/
- **ADNI**: https://adni.loni.usc.edu/
- **UK Biobank**: https://www.ukbiobank.ac.uk/
- **ELSA**: https://www.elsa-project.ac.uk/

## Usage

### 1. Data Preprocessing
```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(config_path="configs/config.yaml")
processed_data = preprocessor.process_all_sources()
```

### 2. Causal Effect Estimation
```python
from src.causal.estimators import CausalEstimator

estimator = CausalEstimator(method="doubly_robust")
cate = estimator.estimate_heterogeneous_effects(X, T, Y)
```

### 3. Policy Learning
```python
from src.policy.optimizer import PolicyOptimizer

optimizer = PolicyOptimizer(constraints={"adherence": 0.7})
policy = optimizer.learn_optimal_policy(X, cate)
```

### 4. Generate Recommendations
```python
recommendation = policy.recommend(patient_features)
print(f"Recommended diet: {recommendation['diet_type']}")
print(f"Expected risk reduction: {recommendation['risk_reduction']:.2%}")
```

### 5. Deploy API
```bash
cd src/api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Launch Web App
```bash
streamlit run src/api/app.py
```

## Methodology
1. **Data Collection & Linkage**: Multi-cohort harmonization
2. **Preprocessing**: FFQ processing, imputation, feature engineering
3. **Causal Framing**: Target trial emulation, confounding adjustment
4. **HTE Estimation**: Causal forests, meta-learners, survival models
5. **Policy Learning**: Contextual bandits with constraints
6. **Validation**: Cross-validation, sensitivity analyses
7. **Deployment**: API + clinical decision support tool

## Expected Outputs
- Personalized treatment rule π(x) with confidence intervals
- Population-level prevented dementia case estimates
- Interpretability reports (SHAP, PDP, subgroup analyses)
- Clinical decision support application
- Validation reports and sensitivity analyses

## Tech Stack
- **Languages**: Python, R
- **Core**: pandas, numpy, scikit-learn, xgboost, lightgbm
- **Causal/HTE**: econml, causalml, dowhy
- **Survival**: lifelines, scikit-survival
- **Optimization**: cvxpy
- **Explainability**: shap, alibi, pdpbox
- **Deployment**: FastAPI, Streamlit, Docker, MLflow

## License
MIT License

## Citation
If you use this code, please cite:
```
@software{dementia_nutrition_ml,
  title={Machine Learning-Assisted Optimization of Dietary Intervention Against Dementia Risk},
  year={2025},
  url={https://github.com/yourusername/dementia_nutrition_ml}
}
```

## Contact
For questions or collaboration, please open an issue on GitHub.
