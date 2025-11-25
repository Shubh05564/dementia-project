"""
Survival analysis for time-to-dementia outcomes.
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
import logging

logger = logging.getLogger(__name__)


class SurvivalAnalyzer:
    """Survival analysis for dementia risk prediction."""
    
    def __init__(self, model_type: str = 'cox_ph'):
        """Initialize survival analyzer.
        
        Args:
            model_type: 'cox_ph' or 'random_survival_forest'
        """
        self.model_type = model_type
        self.model = self._init_model()
        
    def _init_model(self):
        """Initialize survival model."""
        if self.model_type == 'cox_ph':
            return CoxPHFitter()
        elif self.model_type == 'random_survival_forest':
            return RandomSurvivalForest(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        T: pd.Series, 
        E: pd.Series,
        duration_col: str = 'duration',
        event_col: str = 'event'
    ):
        """Fit survival model.
        
        Args:
            X: Covariates
            T: Time to event
            E: Event indicator (1 = event, 0 = censored)
            duration_col: Name for duration column
            event_col: Name for event column
        """
        logger.info(f"Fitting {self.model_type} model...")
        
        if self.model_type == 'cox_ph':
            # Prepare dataframe for lifelines
            df = X.copy()
            df[duration_col] = T
            df[event_col] = E
            
            self.model.fit(df, duration_col=duration_col, event_col=event_col)
            
            logger.info("Cox PH model fitted")
            logger.info(f"Concordance index: {self.model.concordance_index_:.4f}")
            
        elif self.model_type == 'random_survival_forest':
            # Convert to structured array for scikit-survival
            y = np.array(
                [(bool(e), t) for e, t in zip(E, T)],
                dtype=[('event', bool), ('time', float)]
            )
            
            self.model.fit(X, y)
            
            # Calculate C-index
            from sksurv.metrics import concordance_index_censored
            pred = self.model.predict(X)
            c_index = concordance_index_censored(E.astype(bool), T, pred)[0]
            
            logger.info("Random Survival Forest fitted")
            logger.info(f"Concordance index: {c_index:.4f}")
        
        return self
    
    def predict_survival_function(self, X: pd.DataFrame, times: np.ndarray = None):
        """Predict survival function for new observations."""
        if self.model_type == 'cox_ph':
            return self.model.predict_survival_function(X, times=times)
        elif self.model_type == 'random_survival_forest':
            return self.model.predict_survival_function(X)
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk scores."""
        if self.model_type == 'cox_ph':
            return self.model.predict_partial_hazard(X).values
        elif self.model_type == 'random_survival_forest':
            return self.model.predict(X)
    
    def get_hazard_ratios(self) -> pd.DataFrame:
        """Get hazard ratios (Cox PH only)."""
        if self.model_type != 'cox_ph':
            raise ValueError("Hazard ratios only available for Cox PH model")
        
        return self.model.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    
    def plot_coefficients(self):
        """Plot model coefficients."""
        if self.model_type == 'cox_ph':
            return self.model.plot()


class KaplanMeierAnalyzer:
    """Kaplan-Meier survival curves and comparisons."""
    
    def __init__(self):
        """Initialize KM analyzer."""
        self.kmf = KaplanMeierFitter()
        self.fitted_groups = {}
        
    def fit(self, T: pd.Series, E: pd.Series, label: str = "All"):
        """Fit Kaplan-Meier model."""
        self.kmf.fit(T, E, label=label)
        self.fitted_groups[label] = (T, E)
        
        logger.info(f"KM curve fitted for {label}")
        logger.info(f"Median survival time: {self.kmf.median_survival_time_:.2f}")
        
        return self
    
    def fit_by_group(
        self, 
        T: pd.Series, 
        E: pd.Series, 
        groups: pd.Series
    ):
        """Fit separate KM curves for each group."""
        for group_name in groups.unique():
            mask = groups == group_name
            T_group = T[mask]
            E_group = E[mask]
            
            kmf_group = KaplanMeierFitter()
            kmf_group.fit(T_group, E_group, label=str(group_name))
            
            self.fitted_groups[str(group_name)] = (T_group, E_group)
            
            logger.info(f"KM curve for {group_name}: median={kmf_group.median_survival_time_:.2f}")
        
        return self
    
    def compare_groups(
        self, 
        T: pd.Series, 
        E: pd.Series, 
        groups: pd.Series
    ) -> Dict:
        """Compare survival curves between groups using log-rank test."""
        group_labels = groups.unique()
        
        if len(group_labels) != 2:
            raise ValueError("Log-rank test requires exactly 2 groups")
        
        mask_1 = groups == group_labels[0]
        mask_2 = groups == group_labels[1]
        
        result = logrank_test(
            T[mask_1], T[mask_2],
            E[mask_1], E[mask_2]
        )
        
        comparison = {
            'group_1': group_labels[0],
            'group_2': group_labels[1],
            'test_statistic': result.test_statistic,
            'p_value': result.p_value,
            'significant': result.p_value < 0.05
        }
        
        logger.info(f"Log-rank test: p={comparison['p_value']:.4f}")
        
        return comparison
    
    def plot(self):
        """Plot survival curves."""
        return self.kmf.plot_survival_function()


class SurvivalCATEEstimator:
    """Estimate conditional treatment effects for survival outcomes."""
    
    def __init__(self):
        """Initialize survival CATE estimator."""
        self.cox_treated = CoxPHFitter()
        self.cox_control = CoxPHFitter()
        
    def fit(
        self, 
        X: pd.DataFrame, 
        T_treatment: pd.Series, 
        T_time: pd.Series, 
        E: pd.Series
    ):
        """Fit separate Cox models for treated and control groups.
        
        Args:
            X: Covariates
            T_treatment: Treatment indicator
            T_time: Time to event
            E: Event indicator
        """
        logger.info("Fitting survival CATE models...")
        
        # Prepare data for treated group
        X_treated = X[T_treatment == 1].copy()
        X_treated['duration'] = T_time[T_treatment == 1]
        X_treated['event'] = E[T_treatment == 1]
        
        # Prepare data for control group
        X_control = X[T_treatment == 0].copy()
        X_control['duration'] = T_time[T_treatment == 0]
        X_control['event'] = E[T_treatment == 0]
        
        # Fit models
        self.cox_treated.fit(X_treated, duration_col='duration', event_col='event')
        self.cox_control.fit(X_control, duration_col='duration', event_col='event')
        
        logger.info("Survival CATE models fitted")
        
        return self
    
    def estimate_cate(self, X: pd.DataFrame, time_horizon: float = 10.0) -> np.ndarray:
        """Estimate CATE as difference in survival probabilities.
        
        Args:
            X: Covariates
            time_horizon: Time horizon for survival probability (years)
        
        Returns:
            CATE as difference in survival probability at time horizon
        """
        # Predict survival functions
        surv_treated = self.cox_treated.predict_survival_function(X, times=[time_horizon])
        surv_control = self.cox_control.predict_survival_function(X, times=[time_horizon])
        
        # CATE = difference in survival probability
        cate = (surv_treated.values[0] - surv_control.values[0])
        
        logger.info(f"Survival CATE at {time_horizon} years: mean={cate.mean():.4f}")
        
        return cate


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate synthetic survival data
    n = 1000
    X, _ = make_classification(n_samples=n, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(10)])
    
    # Simulate time and event
    T = np.random.exponential(scale=5, size=n)
    E = np.random.binomial(1, 0.7, size=n)
    
    # Fit Cox PH model
    analyzer = SurvivalAnalyzer(model_type='cox_ph')
    analyzer.fit(X_df, pd.Series(T), pd.Series(E))
    
    # Get hazard ratios
    hr = analyzer.get_hazard_ratios()
    print(hr.head())
