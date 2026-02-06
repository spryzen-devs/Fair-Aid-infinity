import pandas as pd
import numpy as np
from typing import Dict

class NormalizationLayer:
    """
    Handles data normalization, imputation, and robustness checks.
    """
    
    @staticmethod
    def process(df: pd.DataFrame, role_mapping: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Applies normalization to the columns identified in the mapping.
        Returns a processed DataFrame with standardized column names (roles) added.
        """
        processed_df = df.copy()
        
        for role, info in role_mapping.items():
            original_col = info['column']
            
            if original_col not in processed_df.columns:
                continue
                
            # Create a standardized column for the role
            standard_col = f"norm_{role.lower()}"
            series = processed_df[original_col]
            
            # Numeric processing
            if pd.api.types.is_numeric_dtype(series):
                # 1. Impute missing (Median)
                if series.hasnans:
                    series = series.fillna(series.median())
                
                # 2. Logic for specific roles
                if role == "AID":
                     # Log scale for large monetary values to reduce skew
                    if series.max() > 10000 and series.min() >= 0:
                         # log(x+1) to handle zeros
                        series = np.log1p(series)
                
                elif role in ["OUTCOME", "POVERTY", "RISK"]:
                    # Normalize percentages to 0-1 if they look like 0-100
                    if series.max() > 1.0 and series.max() <= 100:
                        series = series / 100.0
                    # Cap at 0-1 if it's supposed to be a rate
                    series = series.clip(0, 1)

            processed_df[standard_col] = series
            
        return processed_df
