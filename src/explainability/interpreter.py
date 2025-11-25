"""
Model interpretability and explainability tools.
Implements SHAP, PDP, and subgroup analysis.
"""

import pandas as pd
import numpy as np
import shap
from pdpbox import pdp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """Interpret ML models for causal effects and predictions."""
    
    def __init__(self, model, X_train: pd.DataFrame):
        """Initialize interpreter.
        
        Args:
            model: Fitted ML model
            X_train: Training data for background distribution
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        method: str = 'tree'
    ) -> np.ndarray:
        """Compute SHAP values for feature importance.
        
        Args:
            X: Data to explain
            method: SHAP method ('tree', 'kernel', 'linear')
        
        Returns:
            SHAP values array
        """
        logger.info(f"Computing SHAP values using {method} explainer...")
        
        if method == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif method == 'kernel':
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                shap.sample(self.X_train, 100)
            )
        elif method == 'linear':
            self.explainer = shap.LinearExplainer(
                self.model,
                self.X_train
            )
        else:
            raise ValueError(f"Unknown SHAP method: {method}")
        
        self.shap_values = self.explainer.shap_values(X)
        
        logger.info(f"SHAP values computed for {len(X)} samples")
        return self.shap_values
    
    def plot_shap_summary(self, X: pd.DataFrame, max_display: int = 20):
        """Plot SHAP summary (feature importance)."""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        shap.summary_plot(self.shap_values, X, max_display=max_display)
        
    def plot_shap_waterfall(self, X: pd.DataFrame, sample_idx: int = 0):
        """Plot SHAP waterfall for individual prediction."""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=X.iloc[sample_idx],
                feature_names=X.columns.tolist()
            )
        )
    
    def get_top_features(
        self,
        X: pd.DataFrame,
        n_features: int = 10
    ) -> pd.DataFrame:
        """Get top features by SHAP importance.
        
        Args:
            X: Data
            n_features: Number of top features
        
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Calculate mean absolute SHAP values
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': shap_importance
        }).sort_values('importance', ascending=False).head(n_features)
        
        return importance_df
    
    def explain_individual(
        self,
        patient_features: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Explain prediction for individual patient.
        
        Args:
            patient_features: Single patient features (1 row)
            feature_names: List of feature names to explain
        
        Returns:
            Explanation dictionary
        """
        if self.shap_values is None:
            self.compute_shap_values(patient_features)
        
        # Get SHAP values for this patient
        patient_shap = self.shap_values[0] if len(self.shap_values.shape) > 1 else self.shap_values
        
        # Get top positive and negative contributions
        shap_df = pd.DataFrame({
            'feature': patient_features.columns,
            'value': patient_features.iloc[0].values,
            'shap_value': patient_shap
        })
        
        top_positive = shap_df.nlargest(5, 'shap_value')
        top_negative = shap_df.nsmallest(5, 'shap_value')
        
        explanation = {
            'prediction': self.model.predict(patient_features)[0],
            'top_positive_factors': top_positive.to_dict('records'),
            'top_negative_factors': top_negative.to_dict('records'),
            'base_value': self.explainer.expected_value
        }
        
        return explanation


class PDPAnalyzer:
    """Partial Dependence Plot analysis."""
    
    def __init__(self, model, X_train: pd.DataFrame):
        """Initialize PDP analyzer.
        
        Args:
            model: Fitted model
            X_train: Training data
        """
        self.model = model
        self.X_train = X_train
        
    def compute_pdp(
        self,
        feature: str,
        num_grid_points: int = 20
    ):
        """Compute Partial Dependence Plot for a feature.
        
        Args:
            feature: Feature name
            num_grid_points: Number of grid points
        
        Returns:
            PDP isolate object
        """
        logger.info(f"Computing PDP for {feature}...")
        
        pdp_iso = pdp.pdp_isolate(
            model=self.model,
            dataset=self.X_train,
            model_features=self.X_train.columns.tolist(),
            feature=feature,
            num_grid_points=num_grid_points
        )
        
        return pdp_iso
    
    def plot_pdp(self, feature: str):
        """Plot PDP for a feature."""
        pdp_iso = self.compute_pdp(feature)
        
        fig, ax = pdp.pdp_plot(pdp_iso, feature, plot_lines=True)
        plt.title(f'Partial Dependence Plot: {feature}')
        return fig
    
    def compute_pdp_interact(
        self,
        features: List[str]
    ):
        """Compute interaction PDP for two features.
        
        Args:
            features: List of exactly 2 feature names
        
        Returns:
            PDP interact object
        """
        if len(features) != 2:
            raise ValueError("Exactly 2 features required for interaction PDP")
        
        logger.info(f"Computing interaction PDP for {features}...")
        
        pdp_interact = pdp.pdp_interact(
            model=self.model,
            dataset=self.X_train,
            model_features=self.X_train.columns.tolist(),
            features=features
        )
        
        return pdp_interact
    
    def plot_pdp_interact(self, features: List[str]):
        """Plot interaction PDP."""
        pdp_interact = self.compute_pdp_interact(features)
        
        fig, ax = pdp.pdp_interact_plot(
            pdp_interact,
            features,
            plot_type='contour'
        )
        return fig


class SubgroupAnalyzer:
    """Analyze treatment effects in population subgroups."""
    
    def __init__(self):
        """Initialize subgroup analyzer."""
        pass
    
    def analyze_by_subgroup(
        self,
        X: pd.DataFrame,
        cate: np.ndarray,
        subgroup_var: str
    ) -> pd.DataFrame:
        """Analyze CATE by subgroups.
        
        Args:
            X: Features
            cate: Conditional treatment effects
            subgroup_var: Variable to define subgroups
        
        Returns:
            DataFrame with subgroup statistics
        """
        logger.info(f"Analyzing subgroups by {subgroup_var}...")
        
        df = X.copy()
        df['cate'] = cate
        
        # Group by subgroup variable
        subgroup_stats = df.groupby(subgroup_var)['cate'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        subgroup_stats.columns = [
            'n', 'mean_cate', 'std_cate', 'min_cate', 'max_cate'
        ]
        
        logger.info(f"Subgroup analysis:\n{subgroup_stats}")
        
        return subgroup_stats
    
    def identify_high_responders(
        self,
        X: pd.DataFrame,
        cate: np.ndarray,
        threshold_percentile: float = 75
    ) -> pd.DataFrame:
        """Identify characteristics of high treatment responders.
        
        Args:
            X: Features
            cate: Conditional treatment effects
            threshold_percentile: Percentile threshold for high responders
        
        Returns:
            Comparison of high vs low responders
        """
        threshold = np.percentile(cate, threshold_percentile)
        
        high_responders = cate >= threshold
        
        comparison = {}
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:
                # Numeric feature
                high_mean = X.loc[high_responders, col].mean()
                low_mean = X.loc[~high_responders, col].mean()
                
                comparison[col] = {
                    'high_responders_mean': high_mean,
                    'low_responders_mean': low_mean,
                    'difference': high_mean - low_mean
                }
        
        comparison_df = pd.DataFrame(comparison).T
        comparison_df['abs_difference'] = comparison_df['difference'].abs()
        comparison_df = comparison_df.sort_values('abs_difference', ascending=False)
        
        logger.info(f"Top differentiating features:\n{comparison_df.head(10)}")
        
        return comparison_df
    
    def plot_subgroup_effects(
        self,
        X: pd.DataFrame,
        cate: np.ndarray,
        subgroup_var: str
    ):
        """Plot CATE distribution by subgroups."""
        df = X.copy()
        df['cate'] = cate
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for subgroup in df[subgroup_var].unique():
            subset = df[df[subgroup_var] == subgroup]['cate']
            ax.hist(subset, alpha=0.5, label=f'{subgroup_var}={subgroup}', bins=30)
        
        ax.set_xlabel('Conditional Average Treatment Effect (CATE)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'CATE Distribution by {subgroup_var}')
        ax.legend()
        
        return fig


class ExplainabilityReport:
    """Generate comprehensive explainability report."""
    
    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        cate_test: np.ndarray
    ):
        """Initialize report generator.
        
        Args:
            model: Fitted model
            X_train: Training data
            X_test: Test data
            cate_test: Test CATEs
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.cate_test = cate_test
        
        self.interpreter = ModelInterpreter(model, X_train)
        self.pdp_analyzer = PDPAnalyzer(model, X_train)
        self.subgroup_analyzer = SubgroupAnalyzer()
        
    def generate_report(
        self,
        output_dir: str = 'reports/',
        subgroup_vars: Optional[List[str]] = None
    ):
        """Generate full explainability report.
        
        Args:
            output_dir: Directory to save report outputs
            subgroup_vars: Variables for subgroup analysis
        """
        logger.info("Generating explainability report...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. SHAP analysis
        logger.info("Computing SHAP values...")
        self.interpreter.compute_shap_values(self.X_test)
        
        # Save SHAP summary
        plt.figure()
        self.interpreter.plot_shap_summary(self.X_test)
        plt.savefig(f'{output_dir}/shap_summary.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Top features
        top_features = self.interpreter.get_top_features(self.X_test)
        top_features.to_csv(f'{output_dir}/top_features.csv', index=False)
        
        # 2. PDPs for top features
        logger.info("Generating PDPs...")
        for feat in top_features.head(5)['feature']:
            try:
                fig = self.pdp_analyzer.plot_pdp(feat)
                fig.savefig(f'{output_dir}/pdp_{feat}.png', bbox_inches='tight', dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Could not generate PDP for {feat}: {e}")
        
        # 3. Subgroup analysis
        if subgroup_vars:
            logger.info("Analyzing subgroups...")
            for var in subgroup_vars:
                if var in self.X_test.columns:
                    subgroup_stats = self.subgroup_analyzer.analyze_by_subgroup(
                        self.X_test, self.cate_test, var
                    )
                    subgroup_stats.to_csv(f'{output_dir}/subgroup_{var}.csv')
                    
                    # Plot
                    fig = self.subgroup_analyzer.plot_subgroup_effects(
                        self.X_test, self.cate_test, var
                    )
                    fig.savefig(f'{output_dir}/subgroup_plot_{var}.png', bbox_inches='tight', dpi=300)
                    plt.close()
        
        # 4. High responder analysis
        logger.info("Identifying high responders...")
        high_responder_features = self.subgroup_analyzer.identify_high_responders(
            self.X_test, self.cate_test
        )
        high_responder_features.to_csv(f'{output_dir}/high_responder_features.csv')
        
        logger.info(f"Explainability report saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    # Fit model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_df, y)
    
    # Interpret
    interpreter = ModelInterpreter(model, X_df)
    top_features = interpreter.get_top_features(X_df)
    print(top_features)
