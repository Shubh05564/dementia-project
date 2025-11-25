"""
Causal inference estimators for dietary interventions and dementia risk.
Implements propensity score methods, IPW, doubly robust estimation, and HTE.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from typing import Optional, Tuple, Dict
import logging

# Causal ML libraries
from econml.dr import DRLearner
from econml.metalearners import XLearner, TLearner, SLearner
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor

logger = logging.getLogger(__name__)


class PropensityScoreEstimator:
    """Estimate propensity scores for treatment assignment."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """Initialize propensity score estimator.
        
        Args:
            model_type: Type of model ('logistic', 'rf', 'xgboost', 'lightgbm')
        """
        self.model_type = model_type
        self.model = self._init_model()
        self.propensity_scores = None
        
    def _init_model(self):
        """Initialize the propensity score model."""
        if self.model_type == 'logistic':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, T: pd.Series) -> 'PropensityScoreEstimator':
        """Fit propensity score model.
        
        Args:
            X: Covariates
            T: Treatment indicator
        """
        logger.info(f"Fitting propensity score model ({self.model_type})...")
        self.model.fit(X, T)
        
        # Calculate propensity scores
        self.propensity_scores = self.model.predict_proba(X)[:, 1]
        
        logger.info(f"Propensity score range: [{self.propensity_scores.min():.3f}, {self.propensity_scores.max():.3f}]")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict propensity scores for new data."""
        return self.model.predict_proba(X)[:, 1]
    
    def check_overlap(self, T: pd.Series, threshold: float = 0.1) -> Dict:
        """Check propensity score overlap between treatment groups."""
        if self.propensity_scores is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        ps_treated = self.propensity_scores[T == 1]
        ps_control = self.propensity_scores[T == 0]
        
        # Check for extreme propensity scores
        n_extreme_low = np.sum(self.propensity_scores < threshold)
        n_extreme_high = np.sum(self.propensity_scores > (1 - threshold))
        
        overlap_stats = {
            'treated_mean': ps_treated.mean(),
            'control_mean': ps_control.mean(),
            'treated_std': ps_treated.std(),
            'control_std': ps_control.std(),
            'n_extreme_low': n_extreme_low,
            'n_extreme_high': n_extreme_high,
            'pct_extreme': (n_extreme_low + n_extreme_high) / len(self.propensity_scores) * 100
        }
        
        logger.info(f"Overlap check: {overlap_stats['pct_extreme']:.1f}% with extreme propensity scores")
        return overlap_stats
    
    def trim_by_propensity(
        self, 
        X: pd.DataFrame, 
        T: pd.Series, 
        Y: pd.Series,
        lower: float = 0.1,
        upper: float = 0.9
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Trim sample by propensity score range."""
        if self.propensity_scores is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        mask = (self.propensity_scores >= lower) & (self.propensity_scores <= upper)
        
        logger.info(f"Trimming: {mask.sum()} / {len(mask)} samples retained ({mask.mean():.1%})")
        
        return X[mask], T[mask], Y[mask]


class IPWEstimator:
    """Inverse Probability Weighting (IPW) for causal effect estimation."""
    
    def __init__(self, ps_estimator: PropensityScoreEstimator):
        """Initialize IPW estimator with propensity scores."""
        self.ps_estimator = ps_estimator
        self.ate = None
        
    def calculate_weights(self, T: pd.Series) -> np.ndarray:
        """Calculate IPW weights."""
        ps = self.ps_estimator.propensity_scores
        
        # Stabilized weights
        weights = np.where(T == 1, 1 / ps, 1 / (1 - ps))
        
        # Clip extreme weights
        weights = np.clip(weights, 0.1, 10)
        
        return weights
    
    def estimate_ate(self, T: pd.Series, Y: pd.Series) -> float:
        """Estimate Average Treatment Effect (ATE) using IPW."""
        weights = self.calculate_weights(T)
        
        # Weighted means
        y1_weighted = np.average(Y[T == 1], weights=weights[T == 1])
        y0_weighted = np.average(Y[T == 0], weights=weights[T == 0])
        
        self.ate = y1_weighted - y0_weighted
        
        logger.info(f"IPW ATE: {self.ate:.4f}")
        return self.ate
    
    def estimate_ate_with_ci(
        self, 
        T: pd.Series, 
        Y: pd.Series,
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        """Estimate ATE with confidence intervals using bootstrap."""
        weights = self.calculate_weights(T)
        
        ate_bootstrap = []
        n = len(T)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            T_boot = T.iloc[idx]
            Y_boot = Y.iloc[idx]
            w_boot = weights[idx]
            
            # Calculate ATE for bootstrap sample
            y1_boot = np.average(Y_boot[T_boot == 1], weights=w_boot[T_boot == 1])
            y0_boot = np.average(Y_boot[T_boot == 0], weights=w_boot[T_boot == 0])
            ate_bootstrap.append(y1_boot - y0_boot)
        
        # Calculate confidence interval
        ate_ci_lower = np.percentile(ate_bootstrap, alpha/2 * 100)
        ate_ci_upper = np.percentile(ate_bootstrap, (1 - alpha/2) * 100)
        
        logger.info(f"IPW ATE: {self.ate:.4f} ({ate_ci_lower:.4f}, {ate_ci_upper:.4f})")
        
        return self.ate, ate_ci_lower, ate_ci_upper


class DoublyRobustEstimator:
    """Doubly Robust (DR) estimator combining propensity scores and outcome regression."""
    
    def __init__(
        self,
        ps_estimator: PropensityScoreEstimator,
        outcome_model_type: str = 'lightgbm'
    ):
        """Initialize DR estimator.
        
        Args:
            ps_estimator: Fitted propensity score estimator
            outcome_model_type: Type of outcome regression model
        """
        self.ps_estimator = ps_estimator
        self.outcome_model_type = outcome_model_type
        self.outcome_model_1 = self._init_outcome_model()  # E[Y|X,T=1]
        self.outcome_model_0 = self._init_outcome_model()  # E[Y|X,T=0]
        self.ate = None
        
    def _init_outcome_model(self):
        """Initialize outcome regression model."""
        if self.outcome_model_type == 'xgboost':
            return xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif self.outcome_model_type == 'lightgbm':
            return lgb.LGBMRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown outcome model: {self.outcome_model_type}")
    
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        """Fit outcome regression models."""
        logger.info("Fitting outcome regression models...")
        
        # Fit model for treated
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        self.outcome_model_1.fit(X_treated, Y_treated)
        
        # Fit model for control
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        self.outcome_model_0.fit(X_control, Y_control)
        
        return self
    
    def estimate_ate(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> float:
        """Estimate ATE using doubly robust estimator."""
        # Get propensity scores
        ps = self.ps_estimator.propensity_scores
        
        # Get predicted outcomes
        mu_1 = self.outcome_model_1.predict(X)
        mu_0 = self.outcome_model_0.predict(X)
        
        # Doubly robust estimator
        dr_1 = mu_1 + (T * (Y - mu_1)) / ps
        dr_0 = mu_0 + ((1 - T) * (Y - mu_0)) / (1 - ps)
        
        self.ate = (dr_1 - dr_0).mean()
        
        logger.info(f"Doubly Robust ATE: {self.ate:.4f}")
        return self.ate
    
    def estimate_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect (CATE)."""
        mu_1 = self.outcome_model_1.predict(X)
        mu_0 = self.outcome_model_0.predict(X)
        
        cate = mu_1 - mu_0
        
        logger.info(f"CATE range: [{cate.min():.4f}, {cate.max():.4f}]")
        return cate


class HTEEstimator:
    """Heterogeneous Treatment Effect estimation using econml."""
    
    def __init__(self, method: str = 'dr_learner'):
        """Initialize HTE estimator.
        
        Args:
            method: HTE estimation method ('dr_learner', 'x_learner', 't_learner', 's_learner')
        """
        self.method = method
        self.estimator = self._init_estimator()
        
    def _init_estimator(self):
        """Initialize HTE estimator based on method."""
        if self.method == 'dr_learner':
            return DRLearner(
                model_propensity=xgb.XGBClassifier(random_state=42),
                model_regression=lgb.LGBMRegressor(random_state=42),
                model_final=lgb.LGBMRegressor(random_state=42),
                cv=3
            )
        elif self.method == 'x_learner':
            return XLearner(
                models=lgb.LGBMRegressor(random_state=42),
                propensity_model=xgb.XGBClassifier(random_state=42),
                cate_models=lgb.LGBMRegressor(random_state=42)
            )
        elif self.method == 't_learner':
            return TLearner(models=[
                lgb.LGBMRegressor(random_state=42),
                lgb.LGBMRegressor(random_state=42)
            ])
        elif self.method == 's_learner':
            return SLearner(overall_model=lgb.LGBMRegressor(random_state=42))
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        """Fit HTE estimator."""
        logger.info(f"Fitting HTE estimator ({self.method})...")
        
        # Convert to numpy arrays
        X_np = X.values
        T_np = T.values
        Y_np = Y.values
        
        self.estimator.fit(Y_np, T_np, X=X_np)
        
        logger.info("HTE estimator fitted successfully")
        return self
    
    def estimate_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate CATE for given covariates."""
        X_np = X.values
        cate = self.estimator.effect(X_np)
        
        logger.info(f"CATE estimated for {len(X)} samples")
        logger.info(f"CATE: mean={cate.mean():.4f}, std={cate.std():.4f}")
        
        return cate
    
    def estimate_ate(self, X: pd.DataFrame) -> float:
        """Estimate ATE as average of CATE."""
        cate = self.estimate_cate(X)
        ate = cate.mean()
        
        logger.info(f"ATE (from CATE): {ate:.4f}")
        return ate


class CausalEstimator:
    """Unified interface for causal inference methods."""
    
    def __init__(
        self,
        method: str = 'doubly_robust',
        ps_model: str = 'xgboost',
        outcome_model: str = 'lightgbm'
    ):
        """Initialize causal estimator.
        
        Args:
            method: Causal estimation method
            ps_model: Propensity score model type
            outcome_model: Outcome regression model type
        """
        self.method = method
        self.ps_model = ps_model
        self.outcome_model = outcome_model
        
        self.ps_estimator = None
        self.estimator = None
        
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        """Fit causal estimator."""
        logger.info(f"Fitting causal estimator (method={self.method})...")
        
        # Fit propensity score model
        self.ps_estimator = PropensityScoreEstimator(model_type=self.ps_model)
        self.ps_estimator.fit(X, T)
        
        # Check overlap
        self.ps_estimator.check_overlap(T)
        
        # Fit causal estimator based on method
        if self.method == 'propensity_score':
            self.estimator = IPWEstimator(self.ps_estimator)
        
        elif self.method == 'ipw':
            self.estimator = IPWEstimator(self.ps_estimator)
        
        elif self.method == 'doubly_robust':
            self.estimator = DoublyRobustEstimator(
                self.ps_estimator,
                outcome_model_type=self.outcome_model
            )
            self.estimator.fit(X, T, Y)
        
        elif self.method in ['dr_learner', 'x_learner', 't_learner', 's_learner']:
            self.estimator = HTEEstimator(method=self.method)
            self.estimator.fit(X, T, Y)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info("Causal estimator fitted successfully")
        return self
    
    def estimate_ate(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> float:
        """Estimate Average Treatment Effect."""
        if self.method in ['propensity_score', 'ipw']:
            return self.estimator.estimate_ate(T, Y)
        elif self.method == 'doubly_robust':
            return self.estimator.estimate_ate(X, T, Y)
        else:
            return self.estimator.estimate_ate(X)
    
    def estimate_heterogeneous_effects(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate heterogeneous (conditional) treatment effects."""
        if self.method == 'doubly_robust':
            return self.estimator.estimate_cate(X)
        elif self.method in ['dr_learner', 'x_learner', 't_learner', 's_learner']:
            return self.estimator.estimate_cate(X)
        else:
            raise ValueError(f"Method {self.method} does not support CATE estimation")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, T = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    Y = T * 0.5 + np.random.randn(1000) * 0.1 + X[:, 0] * 0.3
    
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(20)])
    T_series = pd.Series(T)
    Y_series = pd.Series(Y)
    
    # Fit causal estimator
    estimator = CausalEstimator(method='doubly_robust')
    estimator.fit(X_df, T_series, Y_series)
    
    # Estimate ATE
    ate = estimator.estimate_ate(X_df, T_series, Y_series)
    print(f"ATE: {ate:.4f}")
    
    # Estimate CATE
    cate = estimator.estimate_heterogeneous_effects(X_df)
    print(f"CATE range: [{cate.min():.4f}, {cate.max():.4f}]")
