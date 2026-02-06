import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class NeedEngine:
    """
    Calculates a composite need score based on SECTOR-SPECIFIC formulas.
    Accepts the mapped column configuration to find the right data.
    """
    
    @staticmethod
    def calculate_need(df: pd.DataFrame, mapping: dict, sector: str) -> pd.DataFrame:
        """
        Calculates composite need score using the finalized mapping and sector rules.
        """
        df = df.copy()
        
        # Helper to get series
        def get_series(concept):
            meta = mapping.get(concept)
            if meta and meta.get('column') in df.columns:
                return df[meta['column']].fillna(0) # Safety fill
            return pd.Series([0]*len(df))

        # 1. Fetch Core Indicators
        poverty = get_series("POVERTY")
        
        # 2. Normalize Inputs (0-1)
        # We normalize locally for the score calculation
        scaler = MinMaxScaler()
        
        def norm(series):
            if series.max() == series.min(): return series # Avoid div by zero
            return (series - series.min()) / (series.max() - series.min())
            
        p_norm = norm(poverty)
        
        # 3. Apply Sector Formulas
        
        if sector == "Education":
            # Formula: 0.5 * Poverty + 0.5 * Dropout
            dropout = get_series("DROPOUT")
            d_norm = norm(dropout)
            df['need_score'] = (0.5 * p_norm) + (0.5 * d_norm)
            
        elif sector == "Health":
            # Formula: 0.4 * Poverty + 0.4 * Disease + 0.2 * (1 - Access)
            # Note: We need to check if 'HEALTH_ACCESS' fits our schema. 
            # In our SemanticMapper, we mapped 'DISEASE_BURDEN'. 
            # For MVP simplicity, let's use: 0.5 * Poverty + 0.5 * Disease
            disease = get_series("DISEASE_BURDEN")
            dis_norm = norm(disease)
            df['need_score'] = (0.5 * p_norm) + (0.5 * dis_norm)
            
        elif sector == "Food Security":
            # Formula: 0.5 * Poverty + 0.5 * Malnutrition
            mal = get_series("MALNUTRITION")
            mal_norm = norm(mal)
            df['need_score'] = (0.5 * p_norm) + (0.5 * mal_norm)
            
        elif sector == "Disaster Relief":
             # Formula: 0.6 * Damage + 0.4 * Poverty
             damage = get_series("DAMAGE_SEVERITY")
             dam_norm = norm(damage)
             df['need_score'] = (0.4 * p_norm) + (0.6 * dam_norm)
             
        else:
            # Fallback
            df['need_score'] = p_norm
            
        return df
