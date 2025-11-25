# Quick Start Guide

## Machine Learning-Assisted Optimization of Dietary Intervention Against Dementia Risk

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd dementia_nutrition_ml
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Project Structure

```
dementia_nutrition_ml/
├── src/
│   ├── data/              # Data preprocessing & harmonization
│   │   ├── preprocessor.py
│   │   └── loader.py
│   ├── causal/            # Causal inference methods
│   │   ├── estimators.py
│   │   └── survival.py
│   ├── policy/            # Policy learning & optimization
│   │   └── optimizer.py
│   ├── explainability/    # Interpretability tools
│   │   └── interpreter.py
│   ├── models/            # Evaluation framework
│   │   └── evaluator.py
│   └── api/               # Deployment
│       ├── main.py        # FastAPI backend
│       └── app.py         # Streamlit frontend
├── notebooks/             # Example notebooks
├── configs/               # Configuration files
├── data/                  # Data storage
├── tests/                 # Unit tests
└── requirements.txt       # Dependencies
```

### Usage

#### 1. Data Preprocessing

```python
from src.data.preprocessor import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(config_path='configs/config.yaml')

# Process data from multiple sources
data_paths = {
    'nhanes': 'data/raw/nhanes_data.csv',
    'adni': 'data/raw/adni_data.csv',
}

processed_data = preprocessor.process_all_sources(data_paths)
preprocessor.save_processed_data(processed_data, 'data/processed/combined_data.parquet')
```

#### 2. Causal Effect Estimation

```python
from src.causal.estimators import CausalEstimator
from src.data.loader import DataLoader

# Load data
loader = DataLoader()
df = loader.load_processed_data()
X, Y, T = loader.split_features_outcome(df, outcome_col='incident_dementia', treatment_col='high_diet_score')

# Fit causal model
estimator = CausalEstimator(method='doubly_robust')
estimator.fit(X, T, Y)

# Estimate effects
ate = estimator.estimate_ate(X, T, Y)
cate = estimator.estimate_heterogeneous_effects(X)

print(f"Average Treatment Effect: {ate:.4f}")
```

#### 3. Policy Learning

```python
from src.policy.optimizer import PolicyOptimizer

# Learn optimal policy with constraints
optimizer = PolicyOptimizer(constraints={'adherence_threshold': 0.7})
policy = optimizer.learn_optimal_policy(X, cate)

# Generate recommendation for a patient
patient_features = pd.DataFrame([{
    'age': 70,
    'bmi': 27,
    'mind_score': 5,
    'apoe_e4_carrier': 1
}])

recommendation = policy.recommend(patient_features, cate_model=estimator)
print(recommendation)
```

#### 4. Model Interpretation

```python
from src.explainability.interpreter import ModelInterpreter

# Initialize interpreter
interpreter = ModelInterpreter(model=estimator.estimator, X_train=X)

# Compute SHAP values
shap_values = interpreter.compute_shap_values(X)

# Get top features
top_features = interpreter.get_top_features(X, n_features=10)
print(top_features)
```

### Running the Application

#### Option 1: Using Docker Compose (Recommended)

```bash
docker-compose up
```

- API: http://localhost:8000
- Web App: http://localhost:8501

#### Option 2: Run Locally

**Start the API:**
```bash
cd src/api
python main.py
```

**Start the Streamlit app (in another terminal):**
```bash
streamlit run src/api/app.py
```

### API Endpoints

**Get personalized recommendation:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "age": 70,
    "sex": 1,
    "apoe_e4_carrier": 1,
    "mind_score": 5.0
  }'
```

**Population analysis:**
```bash
curl -X POST http://localhost:8000/population-analysis \
  -H "Content-Type: application/json" \
  -d '{"intervention_type": "mind_diet"}'
```

### Accessing Data

Download data from these sources:

1. **NHANES**: https://wwwn.cdc.gov/nchs/nhanes/
2. **ADNI**: https://adni.loni.usc.edu/ (requires registration)
3. **UK Biobank**: https://www.ukbiobank.ac.uk/ (requires approved access)
4. **ELSA**: https://www.elsa-project.ac.uk/ (requires registration)

Place raw data in `data/raw/` directory.

### Example Notebook

See `notebooks/01_example_pipeline.ipynb` for a complete walkthrough.

### Configuration

Edit `configs/config.yaml` to customize:
- Data sources and paths
- Model parameters
- Policy constraints
- Deployment settings

### Testing

```bash
pytest tests/
```

### Key Features

✅ Multi-cohort data harmonization (NHANES, ADNI, UK Biobank, ELSA)  
✅ Causal inference (propensity scores, IPW, doubly robust)  
✅ Heterogeneous treatment effect estimation  
✅ Survival analysis (Cox PH, Kaplan-Meier)  
✅ Policy optimization with constraints  
✅ Model interpretability (SHAP, PDP, subgroup analysis)  
✅ FastAPI + Streamlit deployment  
✅ Cross-validation and sensitivity analyses  

### Support

For questions or issues:
1. Check the README.md
2. Review example notebook
3. Open an issue on GitHub

### Citation

```bibtex
@software{dementia_nutrition_ml,
  title={Machine Learning-Assisted Optimization of Dietary Intervention Against Dementia Risk},
  year={2025},
  url={https://github.com/yourusername/dementia_nutrition_ml}
}
```

---

**Note**: This tool is for research and educational purposes. Always consult healthcare professionals before making dietary changes.
