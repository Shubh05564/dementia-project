"""
Data loader module for multi-cohort dementia datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare datasets for causal inference and ML modeling."""
    
    def __init__(self, data_path: str = "data/processed/"):
        """Initialize data loader."""
        self.data_path = Path(data_path)
        
    def load_processed_data(self, filename: str = "combined_data.parquet") -> pd.DataFrame:
        """Load preprocessed data."""
        file_path = self.data_path / filename
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def split_features_outcome(
        self, 
        df: pd.DataFrame, 
        outcome_col: str = 'incident_dementia',
        treatment_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Split data into features (X), outcome (Y), and optionally treatment (T)."""
        
        # Exclude non-feature columns
        exclude_cols = [
            'participant_id', 'source', outcome_col,
            'event_time', 'event_indicator', 'followup_years'
        ]
        
        if treatment_col:
            exclude_cols.append(treatment_col)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        Y = df[outcome_col]
        T = df[treatment_col] if treatment_col else None
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        
        return X, Y, T
    
    def create_treatment_variable(
        self, 
        df: pd.DataFrame, 
        diet_score_col: str = 'mind_score',
        threshold: float = 7.0
    ) -> pd.Series:
        """Create binary treatment variable based on diet adherence.
        
        Args:
            df: DataFrame with diet scores
            diet_score_col: Column name for diet score
            threshold: Threshold for high adherence (default: 7 on MIND scale)
        
        Returns:
            Binary treatment indicator (1 = high adherence, 0 = low adherence)
        """
        treatment = (df[diet_score_col] >= threshold).astype(int)
        logger.info(f"Treatment prevalence: {treatment.mean():.2%}")
        
        return treatment
    
    def filter_eligible_population(
        self, 
        df: pd.DataFrame,
        min_age: int = 50,
        exclude_baseline_dementia: bool = True
    ) -> pd.DataFrame:
        """Filter for eligible study population."""
        
        n_original = len(df)
        
        # Age eligibility
        if 'age' in df.columns:
            df = df[df['age'] >= min_age]
            logger.info(f"After age filter (>={min_age}): {len(df)} rows")
        
        # Exclude baseline dementia
        if exclude_baseline_dementia and 'baseline_dementia' in df.columns:
            df = df[df['baseline_dementia'] == 0]
            logger.info(f"After excluding baseline dementia: {len(df)} rows")
        
        logger.info(f"Filtered from {n_original} to {len(df)} rows ({len(df)/n_original:.1%})")
        
        return df
    
    def prepare_for_survival_analysis(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for survival analysis.
        
        Returns:
            X: Features
            T: Time to event
            E: Event indicator (1 = event occurred, 0 = censored)
        """
        
        exclude_cols = [
            'participant_id', 'source', 
            'time_to_event', 'event_indicator',
            'incident_dementia'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        T = df['time_to_event']
        E = df['event_indicator']
        
        logger.info(f"Survival data prepared: {len(X)} samples, {E.sum()} events")
        
        return X, T, E
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train-test split."""
        from sklearn.model_selection import train_test_split
        
        stratify = df[stratify_col] if stratify_col else None
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        
        return train_df, test_df
    
    def get_feature_groups(self, df: pd.DataFrame) -> dict:
        """Identify feature groups for interpretation."""
        
        feature_groups = {
            'dietary': [],
            'genetic': [],
            'clinical': [],
            'socioeconomic': [],
            'interaction': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(kw in col_lower for kw in ['diet', 'food', 'nutrient', 'mind', 'mediterranean']):
                feature_groups['dietary'].append(col)
            elif any(kw in col_lower for kw in ['apoe', 'genetic', 'gene']):
                feature_groups['genetic'].append(col)
            elif any(kw in col_lower for kw in ['age', 'bmi', 'comorbid', 'medication']):
                feature_groups['clinical'].append(col)
            elif any(kw in col_lower for kw in ['education', 'income', 'employment']):
                feature_groups['socioeconomic'].append(col)
            elif '_x_' in col_lower:
                feature_groups['interaction'].append(col)
        
        return feature_groups
