"""
Model evaluation and validation framework.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CausalModelEvaluator:
    """Evaluate causal inference models."""
    
    def __init__(self, n_folds: int = 5):
        """Initialize evaluator.
        
        Args:
            n_folds: Number of cross-validation folds
        """
        self.n_folds = n_folds
        
    def evaluate_ate_estimation(
        self,
        estimated_ate: float,
        true_ate: Optional[float] = None,
        bootstrap_estimates: Optional[List[float]] = None
    ) -> Dict:
        """Evaluate ATE estimation quality.
        
        Args:
            estimated_ate: Estimated ATE
            true_ate: True ATE (if known, e.g., from simulation)
            bootstrap_estimates: Bootstrap ATE estimates for CI
        
        Returns:
            Evaluation metrics
        """
        metrics = {
            'ate_estimate': estimated_ate
        }
        
        if true_ate is not None:
            metrics['bias'] = estimated_ate - true_ate
            metrics['absolute_bias'] = abs(estimated_ate - true_ate)
            metrics['relative_bias'] = (estimated_ate - true_ate) / true_ate if true_ate != 0 else np.nan
        
        if bootstrap_estimates is not None:
            metrics['ate_std'] = np.std(bootstrap_estimates)
            metrics['ate_ci_lower'] = np.percentile(bootstrap_estimates, 2.5)
            metrics['ate_ci_upper'] = np.percentile(bootstrap_estimates, 97.5)
            metrics['ate_ci_width'] = metrics['ate_ci_upper'] - metrics['ate_ci_lower']
        
        return metrics
    
    def evaluate_cate_estimation(
        self,
        estimated_cate: np.ndarray,
        true_cate: Optional[np.ndarray] = None
    ) -> Dict:
        """Evaluate CATE estimation quality.
        
        Args:
            estimated_cate: Estimated CATEs
            true_cate: True CATEs (if known)
        
        Returns:
            Evaluation metrics
        """
        metrics = {
            'cate_mean': estimated_cate.mean(),
            'cate_std': estimated_cate.std(),
            'cate_min': estimated_cate.min(),
            'cate_max': estimated_cate.max(),
            'cate_q25': np.percentile(estimated_cate, 25),
            'cate_q75': np.percentile(estimated_cate, 75)
        }
        
        if true_cate is not None:
            # RMSE of CATE
            metrics['cate_rmse'] = np.sqrt(np.mean((estimated_cate - true_cate) ** 2))
            metrics['cate_mae'] = np.mean(np.abs(estimated_cate - true_cate))
            metrics['cate_r2'] = 1 - np.sum((estimated_cate - true_cate) ** 2) / np.sum((true_cate - true_cate.mean()) ** 2)
        
        return metrics
    
    def calculate_policy_value(
        self,
        treatment_policy: np.ndarray,
        outcomes: np.ndarray,
        treatment_actual: np.ndarray,
        propensity_scores: Optional[np.ndarray] = None
    ) -> float:
        """Calculate policy value using IPW.
        
        Args:
            treatment_policy: Policy treatment assignments
            outcomes: Observed outcomes
            treatment_actual: Actual treatment assignments
            propensity_scores: Propensity scores (optional)
        
        Returns:
            Policy value estimate
        """
        if propensity_scores is None:
            # Simple on-policy evaluation
            policy_value = outcomes[treatment_policy == treatment_actual].mean()
        else:
            # IPW evaluation
            weights = np.where(
                treatment_actual == treatment_policy,
                1 / propensity_scores,
                1 / (1 - propensity_scores)
            )
            weights = np.clip(weights, 0.1, 10)  # Clip extreme weights
            
            policy_value = np.average(
                outcomes[treatment_policy == treatment_actual],
                weights=weights[treatment_policy == treatment_actual]
            )
        
        return policy_value
    
    def calculate_qini_coefficient(
        self,
        cate: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Qini coefficient for uplift modeling.
        
        Args:
            cate: Estimated CATEs
            treatment: Treatment assignments
            outcome: Outcomes
            n_bins: Number of bins for Qini curve
        
        Returns:
            Qini coefficient
        """
        # Sort by CATE (descending)
        sorted_idx = np.argsort(-cate)
        cate_sorted = cate[sorted_idx]
        treatment_sorted = treatment[sorted_idx]
        outcome_sorted = outcome[sorted_idx]
        
        n = len(cate)
        bin_size = n // n_bins
        
        qini_values = []
        for i in range(1, n_bins + 1):
            idx = min(i * bin_size, n)
            
            treated = treatment_sorted[:idx] == 1
            control = treatment_sorted[:idx] == 0
            
            if treated.sum() > 0 and control.sum() > 0:
                uplift = (outcome_sorted[:idx][treated].mean() - 
                         outcome_sorted[:idx][control].mean())
                qini_values.append(uplift * idx)
            else:
                qini_values.append(0)
        
        # Qini coefficient = area under curve
        qini_coef = np.trapz(qini_values, dx=1) / (n * n_bins)
        
        return qini_coef


class SensitivityAnalyzer:
    """Conduct sensitivity analyses."""
    
    def __init__(self):
        """Initialize sensitivity analyzer."""
        pass
    
    def unmeasured_confounding_sensitivity(
        self,
        ate_estimate: float,
        gamma_range: np.ndarray = np.linspace(1, 3, 10)
    ) -> pd.DataFrame:
        """Sensitivity analysis for unmeasured confounding.
        
        Uses Rosenbaum's Gamma sensitivity analysis approach.
        
        Args:
            ate_estimate: Original ATE estimate
            gamma_range: Range of Gamma values (confounding strength)
        
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        for gamma in gamma_range:
            # Simplified sensitivity bounds
            # In practice, use more sophisticated methods
            ate_upper = ate_estimate * gamma
            ate_lower = ate_estimate / gamma
            
            results.append({
                'gamma': gamma,
                'ate_lower_bound': ate_lower,
                'ate_upper_bound': ate_upper,
                'bound_width': ate_upper - ate_lower
            })
        
        return pd.DataFrame(results)
    
    def missing_data_sensitivity(
        self,
        df: pd.DataFrame,
        missing_scenarios: List[Dict]
    ) -> pd.DataFrame:
        """Sensitivity analysis for missing data mechanisms.
        
        Args:
            df: Dataset
            missing_scenarios: List of missing data scenarios to test
        
        Returns:
            DataFrame with sensitivity results
        """
        logger.info("Conducting missing data sensitivity analysis...")
        
        results = []
        for scenario in missing_scenarios:
            # Implement different missing data mechanisms
            # This is a placeholder for the actual implementation
            results.append({
                'scenario': scenario['name'],
                'missing_pct': scenario.get('missing_pct', 0),
                'ate_estimate': np.nan,  # Would calculate with imputed data
                'bias_estimate': np.nan
            })
        
        return pd.DataFrame(results)


class CrossValidator:
    """Cross-validation for causal models."""
    
    def __init__(self, n_splits: int = 5, stratify: bool = True):
        """Initialize cross-validator.
        
        Args:
            n_splits: Number of CV folds
            stratify: Whether to stratify by outcome
        """
        self.n_splits = n_splits
        self.stratify = stratify
        
    def cross_validate_model(
        self,
        model,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series
    ) -> Dict:
        """Cross-validate causal model.
        
        Args:
            model: Causal model to evaluate
            X: Features
            T: Treatment
            Y: Outcome
        
        Returns:
            Cross-validation results
        """
        logger.info(f"Running {self.n_splits}-fold cross-validation...")
        
        if self.stratify:
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = kf.split(X, Y)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = kf.split(X)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Fold {fold_idx + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            T_train, T_val = T.iloc[train_idx], T.iloc[val_idx]
            Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]
            
            # Fit model
            model.fit(X_train, T_train, Y_train)
            
            # Estimate effects
            ate_val = model.estimate_ate(X_val, T_val, Y_val)
            cate_val = model.estimate_heterogeneous_effects(X_val)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'ate': ate_val,
                'cate_mean': cate_val.mean(),
                'cate_std': cate_val.std()
            })
        
        # Aggregate results
        ate_mean = np.mean([r['ate'] for r in fold_results])
        ate_std = np.std([r['ate'] for r in fold_results])
        
        cv_results = {
            'ate_mean': ate_mean,
            'ate_std': ate_std,
            'ate_cv': ate_std / ate_mean if ate_mean != 0 else np.nan,
            'fold_results': fold_results
        }
        
        logger.info(f"CV ATE: {ate_mean:.4f} ± {ate_std:.4f}")
        
        return cv_results


class ValidationReport:
    """Generate comprehensive validation report."""
    
    def __init__(self):
        """Initialize report generator."""
        self.evaluator = CausalModelEvaluator()
        self.sensitivity = SensitivityAnalyzer()
        self.cv = CrossValidator()
        
    def generate_full_report(
        self,
        model,
        X_train: pd.DataFrame,
        T_train: pd.Series,
        Y_train: pd.Series,
        X_test: pd.DataFrame,
        T_test: pd.Series,
        Y_test: pd.Series,
        output_path: str = 'reports/validation_report.txt'
    ):
        """Generate comprehensive validation report.
        
        Args:
            model: Fitted causal model
            X_train: Training features
            T_train: Training treatment
            Y_train: Training outcome
            X_test: Test features
            T_test: Test treatment
            Y_test: Test outcome
            output_path: Path to save report
        """
        logger.info("Generating validation report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CAUSAL MODEL VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. Cross-validation
        report_lines.append("1. CROSS-VALIDATION RESULTS")
        report_lines.append("-" * 80)
        cv_results = self.cv.cross_validate_model(model, X_train, T_train, Y_train)
        report_lines.append(f"ATE (mean ± std): {cv_results['ate_mean']:.4f} ± {cv_results['ate_std']:.4f}")
        report_lines.append(f"Coefficient of Variation: {cv_results['ate_cv']:.4f}")
        report_lines.append("")
        
        # 2. Test set evaluation
        report_lines.append("2. TEST SET EVALUATION")
        report_lines.append("-" * 80)
        ate_test = model.estimate_ate(X_test, T_test, Y_test)
        cate_test = model.estimate_heterogeneous_effects(X_test)
        
        ate_metrics = self.evaluator.evaluate_ate_estimation(ate_test)
        cate_metrics = self.evaluator.evaluate_cate_estimation(cate_test)
        
        report_lines.append(f"ATE (test): {ate_metrics['ate_estimate']:.4f}")
        report_lines.append(f"CATE mean: {cate_metrics['cate_mean']:.4f}")
        report_lines.append(f"CATE std: {cate_metrics['cate_std']:.4f}")
        report_lines.append(f"CATE range: [{cate_metrics['cate_min']:.4f}, {cate_metrics['cate_max']:.4f}]")
        report_lines.append("")
        
        # 3. Sensitivity analysis
        report_lines.append("3. SENSITIVITY ANALYSIS")
        report_lines.append("-" * 80)
        sensitivity_results = self.sensitivity.unmeasured_confounding_sensitivity(ate_test)
        report_lines.append("Unmeasured confounding sensitivity:")
        for _, row in sensitivity_results.head(5).iterrows():
            report_lines.append(f"  Gamma={row['gamma']:.2f}: ATE bounds [{row['ate_lower_bound']:.4f}, {row['ate_upper_bound']:.4f}]")
        report_lines.append("")
        
        # Save report
        report_text = "\n".join(report_lines)
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Validation report saved to {output_path}")
        
        return report_text


if __name__ == "__main__":
    # Example usage
    from src.causal.estimators import CausalEstimator
    
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        'age': np.random.normal(65, 10, n),
        'bmi': np.random.normal(25, 5, n),
        'mind_score': np.random.uniform(0, 15, n)
    })
    T = (X['mind_score'] > 7).astype(int)
    Y = (0.5 - 0.01 * X['age'] - 0.1 * T + np.random.randn(n) * 0.1 > 0).astype(int)
    
    # Fit model
    model = CausalEstimator(method='doubly_robust')
    model.fit(X, T, Y)
    
    # Evaluate
    evaluator = CausalModelEvaluator()
    ate = model.estimate_ate(X, T, Y)
    cate = model.estimate_heterogeneous_effects(X)
    
    ate_metrics = evaluator.evaluate_ate_estimation(ate)
    cate_metrics = evaluator.evaluate_cate_estimation(cate)
    
    print("ATE metrics:", ate_metrics)
    print("CATE metrics:", cate_metrics)
