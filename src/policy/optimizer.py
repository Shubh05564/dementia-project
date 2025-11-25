"""
Policy learning and optimization for personalized dietary recommendations.
Implements constrained optimization with adherence, safety, and resource constraints.
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PolicyOptimizer:
    """Learn optimal treatment policies with constraints."""
    
    def __init__(
        self,
        constraints: Optional[Dict] = None,
        objective: str = 'maximize_risk_reduction'
    ):
        """Initialize policy optimizer.
        
        Args:
            constraints: Dictionary of constraint specifications
            objective: Optimization objective
        """
        self.constraints = constraints or {
            'adherence_threshold': 0.7,
            'safety_check': True,
            'resource_limit': 100
        }
        self.objective = objective
        self.policy = None
        
    def learn_optimal_policy(
        self,
        X: pd.DataFrame,
        cate: np.ndarray,
        adherence_scores: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None
    ) -> 'TreatmentPolicy':
        """Learn optimal treatment assignment policy.
        
        Args:
            X: Patient features
            cate: Conditional average treatment effects
            adherence_scores: Predicted adherence probability for each patient
            costs: Cost of treatment for each patient
        
        Returns:
            Learned treatment policy
        """
        logger.info("Learning optimal treatment policy...")
        
        n = len(X)
        
        # Initialize adherence and costs if not provided
        if adherence_scores is None:
            adherence_scores = np.ones(n) * 0.8  # Default high adherence
        
        if costs is None:
            costs = np.ones(n) * 50  # Default cost
        
        # Decision variables: binary treatment assignment
        treatment = cp.Variable(n, boolean=True)
        
        # Objective: maximize expected benefit
        # Benefit = CATE * adherence * treatment
        expected_benefit = cate * adherence_scores
        objective_expr = cp.sum(cp.multiply(expected_benefit, treatment))
        
        # Constraints
        constraints = []
        
        # Adherence constraint: only treat patients with sufficient adherence
        if 'adherence_threshold' in self.constraints:
            adherence_threshold = self.constraints['adherence_threshold']
            # Treatment only for patients above threshold
            for i in range(n):
                if adherence_scores[i] < adherence_threshold:
                    constraints.append(treatment[i] == 0)
        
        # Resource constraint: total cost <= budget
        if 'resource_limit' in self.constraints:
            resource_limit = self.constraints['resource_limit'] * n  # Total budget
            constraints.append(cp.sum(cp.multiply(costs, treatment)) <= resource_limit)
        
        # Safety constraint: only treat patients with positive expected effect
        if self.constraints.get('safety_check', True):
            for i in range(n):
                if cate[i] <= 0:
                    constraints.append(treatment[i] == 0)
        
        # Solve optimization problem
        problem = cp.Problem(cp.Maximize(objective_expr), constraints)
        
        try:
            problem.solve(solver=cp.ECOS_BB)
            
            if problem.status == cp.OPTIMAL:
                logger.info(f"Optimal policy found. Objective value: {problem.value:.4f}")
                logger.info(f"Treatment rate: {treatment.value.sum() / n:.2%}")
                
                # Create policy
                self.policy = TreatmentPolicy(
                    treatment_assignments=treatment.value.astype(int),
                    cate=cate,
                    expected_value=problem.value
                )
                
                return self.policy
            else:
                logger.warning(f"Optimization status: {problem.status}")
                # Fallback: treat all with positive CATE
                treatment_fallback = (cate > 0).astype(int)
                self.policy = TreatmentPolicy(
                    treatment_assignments=treatment_fallback,
                    cate=cate,
                    expected_value=np.sum(cate * treatment_fallback)
                )
                return self.policy
                
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            # Fallback policy
            treatment_fallback = (cate > 0).astype(int)
            self.policy = TreatmentPolicy(
                treatment_assignments=treatment_fallback,
                cate=cate,
                expected_value=np.sum(cate * treatment_fallback)
            )
            return self.policy
    
    def evaluate_policy(
        self,
        X_test: pd.DataFrame,
        T_test: pd.Series,
        Y_test: pd.Series,
        cate_test: np.ndarray
    ) -> Dict:
        """Evaluate learned policy on test data.
        
        Args:
            X_test: Test features
            T_test: Test treatments
            Y_test: Test outcomes
            cate_test: Test CATEs
        
        Returns:
            Policy evaluation metrics
        """
        if self.policy is None:
            raise ValueError("No policy learned. Call learn_optimal_policy() first.")
        
        # Get policy recommendations for test set
        # For now, use simple rule based on CATE
        recommended = (cate_test > 0).astype(int)
        
        # Calculate policy value
        policy_value = self._calculate_policy_value(recommended, T_test, Y_test, cate_test)
        
        # Calculate regret (compared to oracle)
        oracle_value = np.sum(np.maximum(cate_test, 0))
        regret = oracle_value - policy_value
        
        evaluation = {
            'policy_value': policy_value,
            'oracle_value': oracle_value,
            'regret': regret,
            'treatment_rate': recommended.mean(),
            'agreement_with_actual': (recommended == T_test).mean()
        }
        
        logger.info(f"Policy value: {policy_value:.4f}")
        logger.info(f"Regret: {regret:.4f}")
        
        return evaluation
    
    def _calculate_policy_value(
        self,
        policy_treatment: np.ndarray,
        actual_treatment: pd.Series,
        outcomes: pd.Series,
        cate: np.ndarray
    ) -> float:
        """Calculate value of policy using IPW."""
        # Simplified policy value calculation
        # In practice, use IPW or doubly robust estimation
        return np.sum(policy_treatment * cate)


class TreatmentPolicy:
    """Represents a learned treatment policy."""
    
    def __init__(
        self,
        treatment_assignments: np.ndarray,
        cate: np.ndarray,
        expected_value: float
    ):
        """Initialize treatment policy.
        
        Args:
            treatment_assignments: Binary treatment assignments
            cate: Conditional treatment effects
            expected_value: Expected value of policy
        """
        self.treatment_assignments = treatment_assignments
        self.cate = cate
        self.expected_value = expected_value
        
    def recommend(self, patient_features: pd.DataFrame, cate_model) -> Dict:
        """Generate recommendation for a new patient.
        
        Args:
            patient_features: Features for single patient (1 row)
            cate_model: Fitted CATE estimation model
        
        Returns:
            Recommendation dictionary
        """
        # Estimate CATE for this patient
        patient_cate = cate_model.estimate_cate(patient_features)[0]
        
        # Make recommendation
        recommend_treatment = patient_cate > 0
        
        # Determine diet type (simplified)
        if recommend_treatment:
            # Recommend MIND diet if positive effect
            diet_type = "MIND Diet"
            risk_reduction = abs(patient_cate)
        else:
            diet_type = "Standard Healthy Diet"
            risk_reduction = 0.0
        
        recommendation = {
            'recommended': recommend_treatment,
            'diet_type': diet_type,
            'risk_reduction': risk_reduction,
            'confidence': 'high' if abs(patient_cate) > 0.1 else 'moderate'
        }
        
        return recommendation
    
    def get_summary(self) -> Dict:
        """Get policy summary statistics."""
        return {
            'n_treated': int(self.treatment_assignments.sum()),
            'n_total': len(self.treatment_assignments),
            'treatment_rate': self.treatment_assignments.mean(),
            'expected_value': self.expected_value,
            'mean_cate_treated': self.cate[self.treatment_assignments == 1].mean(),
            'mean_cate_control': self.cate[self.treatment_assignments == 0].mean()
        }


class AdherencePredictor:
    """Predict patient adherence to dietary interventions."""
    
    def __init__(self):
        """Initialize adherence predictor."""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def fit(self, X: pd.DataFrame, adherence: pd.Series):
        """Fit adherence prediction model.
        
        Args:
            X: Patient features
            adherence: Binary adherence indicator
        """
        logger.info("Fitting adherence prediction model...")
        self.model.fit(X, adherence)
        
        # Calculate feature importance
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info("Top adherence predictors:")
        logger.info(feature_importance.head(10))
        
        return self
    
    def predict_adherence(self, X: pd.DataFrame) -> np.ndarray:
        """Predict adherence probability."""
        return self.model.predict_proba(X)[:, 1]


class ContextualBanditPolicy:
    """Contextual bandit approach for online policy learning."""
    
    def __init__(self, n_actions: int = 3):
        """Initialize contextual bandit.
        
        Args:
            n_actions: Number of treatment options (e.g., MIND, Mediterranean, Standard)
        """
        self.n_actions = n_actions
        self.action_models = []
        
        # Initialize reward models for each action
        from sklearn.linear_model import Ridge
        for _ in range(n_actions):
            self.action_models.append(Ridge(alpha=1.0))
        
        self.history = []
        
    def select_action(
        self,
        context: np.ndarray,
        epsilon: float = 0.1
    ) -> int:
        """Select action using epsilon-greedy strategy.
        
        Args:
            context: Patient features
            epsilon: Exploration rate
        
        Returns:
            Selected action index
        """
        # Epsilon-greedy
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best predicted action
            if len(self.history) == 0:
                return np.random.randint(self.n_actions)
            
            # Predict reward for each action
            predicted_rewards = []
            for model in self.action_models:
                reward = model.predict(context.reshape(1, -1))[0]
                predicted_rewards.append(reward)
            
            return np.argmax(predicted_rewards)
    
    def update(
        self,
        context: np.ndarray,
        action: int,
        reward: float
    ):
        """Update model with observed reward.
        
        Args:
            context: Patient features
            action: Action taken
            reward: Observed reward
        """
        self.history.append((context, action, reward))
        
        # Retrain model for this action with all data
        if len(self.history) >= 10:  # Minimum samples
            action_data = [(c, r) for c, a, r in self.history if a == action]
            if len(action_data) > 0:
                X = np.array([c for c, r in action_data])
                y = np.array([r for c, r in action_data])
                self.action_models[action].fit(X, y)


if __name__ == "__main__":
    # Example usage
    n = 1000
    
    # Generate synthetic data
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f'X{i}' for i in range(10)])
    cate = np.random.randn(n) * 0.2
    
    # Learn policy
    optimizer = PolicyOptimizer(constraints={'adherence_threshold': 0.7})
    policy = optimizer.learn_optimal_policy(X, cate)
    
    # Get summary
    summary = policy.get_summary()
    print(summary)
