"""
Data preprocessing and harmonization module for multi-cohort dementia studies.
Handles NHANES, ADNI, UK Biobank, and ELSA data sources.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
import yaml
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and harmonize multi-source dietary and health data."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        self.imputer = IterativeImputer(
            max_iter=self.config['preprocessing']['n_imputations'],
            random_state=self.config['project']['random_seed']
        )
        self.pca = None
        
    def load_data(self, source: str, file_path: str) -> pd.DataFrame:
        """Load data from specified source."""
        logger.info(f"Loading data from {source}: {file_path}")
        
        # Load based on file format
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Loaded {len(df)} rows from {source}")
        return df
    
    def harmonize_nhanes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize NHANES data to standard format."""
        logger.info("Harmonizing NHANES data...")
        
        # Map NHANES variable names to standard names
        nhanes_mapping = {
            'SEQN': 'participant_id',
            'RIDAGEYR': 'age',
            'RIAGENDR': 'sex',
            'BMXBMI': 'bmi',
            'DMDEDUC2': 'education',
            'INDFMPIR': 'income_poverty_ratio',
            # Add more mappings as needed
        }
        
        df = df.rename(columns=nhanes_mapping)
        df['source'] = 'nhanes'
        
        # Standardize sex coding (1=Male, 2=Female -> 0=Female, 1=Male)
        if 'sex' in df.columns:
            df['sex'] = (df['sex'] == 1).astype(int)
        
        return df
    
    def harmonize_adni(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize ADNI data to standard format."""
        logger.info("Harmonizing ADNI data...")
        
        adni_mapping = {
            'RID': 'participant_id',
            'AGE': 'age',
            'PTGENDER': 'sex',
            'APOE4': 'apoe_e4_status',
            'DX': 'diagnosis',
            'MMSE': 'mmse_score',
            'ADAS13': 'adas_score',
            # Add more mappings
        }
        
        df = df.rename(columns=adni_mapping)
        df['source'] = 'adni'
        
        # Standardize sex coding
        if 'sex' in df.columns:
            df['sex'] = (df['sex'] == 'Male').astype(int)
        
        return df
    
    def harmonize_ukbiobank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize UK Biobank data to standard format."""
        logger.info("Harmonizing UK Biobank data...")
        
        ukb_mapping = {
            'eid': 'participant_id',
            '21022-0.0': 'age',
            '31-0.0': 'sex',
            '21001-0.0': 'bmi',
            # Add UK Biobank field IDs
        }
        
        df = df.rename(columns=ukb_mapping)
        df['source'] = 'ukbiobank'
        
        return df
    
    def harmonize_elsa(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize ELSA data to standard format."""
        logger.info("Harmonizing ELSA data...")
        
        elsa_mapping = {
            'idauniq': 'participant_id',
            'dhager': 'age',
            'dhsex': 'sex',
            # Add more mappings
        }
        
        df = df.rename(columns=elsa_mapping)
        df['source'] = 'elsa'
        
        return df
    
    def process_ffq_to_nutrients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Food Frequency Questionnaire responses to nutrient intake."""
        logger.info("Converting FFQ to nutrient intake...")
        
        # Example nutrient calculation (simplified)
        # In practice, use comprehensive FFQ databases like USDA
        
        nutrient_cols = []
        
        # Calculate major nutrients if food columns exist
        if 'vegetables_servings' in df.columns:
            df['fiber_g'] = df['vegetables_servings'] * 3.5  # Example conversion
            nutrient_cols.append('fiber_g')
        
        if 'fish_servings' in df.columns:
            df['omega3_g'] = df['fish_servings'] * 1.2
            nutrient_cols.append('omega3_g')
        
        # Add more nutrient calculations
        
        return df
    
    def calculate_diet_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MIND and Mediterranean diet scores."""
        logger.info("Calculating diet scores...")
        
        # MIND diet score (0-15 scale)
        # Components: green leafy veg, other veg, nuts, berries, beans, 
        # whole grains, fish, poultry, olive oil, wine, red meat, 
        # butter, cheese, pastries, fried food
        
        mind_components = []
        
        # Beneficial components (1 point each if meets threshold)
        if 'green_leafy_veg' in df.columns:
            mind_components.append((df['green_leafy_veg'] >= 6).astype(int))
        
        if 'nuts_servings' in df.columns:
            mind_components.append((df['nuts_servings'] >= 5).astype(int))
        
        if 'berries_servings' in df.columns:
            mind_components.append((df['berries_servings'] >= 2).astype(int))
        
        if 'fish_servings' in df.columns:
            mind_components.append((df['fish_servings'] >= 1).astype(int))
        
        # Harmful components (1 point if LESS than threshold)
        if 'red_meat_servings' in df.columns:
            mind_components.append((df['red_meat_servings'] < 4).astype(int))
        
        if len(mind_components) > 0:
            df['mind_score'] = sum(mind_components)
        else:
            df['mind_score'] = np.nan
        
        # Mediterranean diet score (simplified)
        med_components = []
        
        if 'olive_oil_use' in df.columns:
            med_components.append(df['olive_oil_use'])
        
        if 'vegetables_servings' in df.columns:
            med_components.append((df['vegetables_servings'] >= 5).astype(int))
        
        if 'fish_servings' in df.columns:
            med_components.append((df['fish_servings'] >= 3).astype(int))
        
        if len(med_components) > 0:
            df['mediterranean_score'] = sum(med_components)
        else:
            df['mediterranean_score'] = np.nan
        
        return df
    
    def create_food_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated food group features."""
        logger.info("Creating food group features...")
        
        # Define food groups
        food_groups = {
            'fruits': ['apple', 'banana', 'orange', 'berries'],
            'vegetables': ['leafy_greens', 'cruciferous', 'root_veg'],
            'whole_grains': ['oats', 'brown_rice', 'quinoa', 'whole_wheat'],
            'legumes': ['beans', 'lentils', 'chickpeas'],
            'nuts_seeds': ['almonds', 'walnuts', 'chia', 'flax'],
            'fish_seafood': ['fatty_fish', 'lean_fish', 'shellfish'],
            'poultry': ['chicken', 'turkey'],
            'red_meat': ['beef', 'pork', 'lamb'],
            'dairy': ['milk', 'yogurt', 'cheese'],
            'processed_foods': ['fast_food', 'packaged_snacks']
        }
        
        for group_name, items in food_groups.items():
            # Sum servings across items in group (if columns exist)
            existing_items = [col for col in items if col in df.columns]
            if existing_items:
                df[f'{group_name}_total'] = df[existing_items].sum(axis=1)
        
        return df
    
    def apply_pca_to_diet(self, df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """Apply PCA to dietary features for dimensionality reduction."""
        logger.info(f"Applying PCA with {n_components} components...")
        
        # Select dietary columns
        diet_cols = [col for col in df.columns if any(
            kw in col.lower() for kw in ['food', 'diet', 'nutrient', 'serving']
        )]
        
        if len(diet_cols) == 0:
            logger.warning("No dietary columns found for PCA")
            return df
        
        # Fit PCA on non-missing data
        diet_data = df[diet_cols].dropna()
        if len(diet_data) > 0:
            self.pca = PCA(n_components=n_components, random_state=self.config['project']['random_seed'])
            pca_features = self.pca.fit_transform(diet_data)
            
            # Add PCA components to dataframe
            pca_df = pd.DataFrame(
                pca_features,
                index=diet_data.index,
                columns=[f'diet_pc_{i+1}' for i in range(n_components)]
            )
            
            df = df.join(pca_df)
            
            logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return df
    
    def encode_apoe_genotype(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode APOE ε4 carrier status."""
        logger.info("Encoding APOE genotype...")
        
        if 'apoe_genotype' in df.columns:
            # Convert genotype to ε4 allele count
            df['apoe_e4_count'] = df['apoe_genotype'].apply(
                lambda x: str(x).count('4') if pd.notna(x) else np.nan
            )
            # Binary carrier status
            df['apoe_e4_carrier'] = (df['apoe_e4_count'] > 0).astype(int)
        
        elif 'apoe_e4_status' not in df.columns:
            # Create placeholder if not available
            df['apoe_e4_carrier'] = np.nan
        
        return df
    
    def impute_missing_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Impute missing values using MICE (Multiple Imputation by Chained Equations)."""
        logger.info("Imputing missing data with MICE...")
        
        if columns is None:
            # Select numeric columns with missing values
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check for columns with too much missingness
        missing_pct = df[columns].isnull().mean()
        cols_to_impute = missing_pct[missing_pct < 0.5].index.tolist()
        
        if len(cols_to_impute) > 0:
            df[cols_to_impute] = self.imputer.fit_transform(df[cols_to_impute])
            logger.info(f"Imputed {len(cols_to_impute)} columns")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Normalize continuous features to zero mean and unit variance."""
        logger.info("Normalizing features...")
        
        if columns is None:
            # Select numeric columns (excluding IDs and binary variables)
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in columns if col not in ['participant_id', 'sex', 'apoe_e4_carrier']]
        
        if len(columns) > 0:
            df[columns] = self.scaler.fit_transform(df[columns])
            logger.info(f"Normalized {len(columns)} columns")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for heterogeneous treatment effects."""
        logger.info("Creating interaction features...")
        
        # Diet × APOE
        if 'mind_score' in df.columns and 'apoe_e4_carrier' in df.columns:
            df['mind_x_apoe'] = df['mind_score'] * df['apoe_e4_carrier']
        
        # Diet × Age
        if 'mediterranean_score' in df.columns and 'age' in df.columns:
            df['med_x_age'] = df['mediterranean_score'] * df['age']
        
        # Diet × Comorbidity (if available)
        if 'mind_score' in df.columns and 'comorbidity_count' in df.columns:
            df['mind_x_comorbidity'] = df['mind_score'] * df['comorbidity_count']
        
        return df
    
    def define_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define primary and secondary outcomes."""
        logger.info("Defining outcomes...")
        
        # Incident dementia (binary)
        if 'dementia_diagnosis' in df.columns:
            df['incident_dementia'] = df['dementia_diagnosis'].astype(int)
        
        # Cognitive decline slope (requires longitudinal data)
        if 'mmse_baseline' in df.columns and 'mmse_followup' in df.columns:
            df['cognitive_decline_slope'] = (
                (df['mmse_followup'] - df['mmse_baseline']) / df['followup_years']
            )
        
        # MCI to AD conversion
        if 'baseline_dx' in df.columns and 'followup_dx' in df.columns:
            df['mci_to_ad_conversion'] = (
                (df['baseline_dx'] == 'MCI') & (df['followup_dx'] == 'AD')
            ).astype(int)
        
        # Time to event (for survival analysis)
        if 'event_time' in df.columns and 'event_occurred' in df.columns:
            df['time_to_event'] = df['event_time']
            df['event_indicator'] = df['event_occurred']
        
        return df
    
    def process_all_sources(self, data_paths: Dict[str, str]) -> pd.DataFrame:
        """Process and combine data from all sources."""
        logger.info("Processing all data sources...")
        
        all_data = []
        
        for source, path in data_paths.items():
            try:
                # Load data
                df = self.load_data(source, path)
                
                # Harmonize based on source
                if source == 'nhanes':
                    df = self.harmonize_nhanes(df)
                elif source == 'adni':
                    df = self.harmonize_adni(df)
                elif source == 'ukbiobank':
                    df = self.harmonize_ukbiobank(df)
                elif source == 'elsa':
                    df = self.harmonize_elsa(df)
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Error processing {source}: {str(e)}")
                continue
        
        # Combine all sources
        if len(all_data) > 0:
            combined_df = pd.concat(all_data, axis=0, ignore_index=True)
            logger.info(f"Combined data: {len(combined_df)} rows from {len(all_data)} sources")
            
            # Apply processing steps
            combined_df = self.process_ffq_to_nutrients(combined_df)
            combined_df = self.calculate_diet_scores(combined_df)
            combined_df = self.create_food_group_features(combined_df)
            combined_df = self.encode_apoe_genotype(combined_df)
            combined_df = self.define_outcomes(combined_df)
            combined_df = self.impute_missing_data(combined_df)
            combined_df = self.create_interaction_features(combined_df)
            combined_df = self.apply_pca_to_diet(
                combined_df, 
                n_components=self.config['preprocessing']['diet_harmonization']['pca_components']
            )
            
            logger.info("Preprocessing complete!")
            return combined_df
        else:
            raise ValueError("No data sources were successfully processed")
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data."""
        logger.info(f"Saving processed data to {output_path}")
        df.to_parquet(output_path, index=False)
        logger.info("Data saved successfully!")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Example data paths (update with actual paths)
    data_paths = {
        'nhanes': 'data/raw/nhanes_data.csv',
        'adni': 'data/raw/adni_data.csv',
    }
    
    # Process data
    # processed_data = preprocessor.process_all_sources(data_paths)
    # preprocessor.save_processed_data(processed_data, 'data/processed/combined_data.parquet')
