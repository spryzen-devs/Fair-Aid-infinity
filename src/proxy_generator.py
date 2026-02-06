import pandas as pd
import numpy as np

class ProxyGenerator:
    """
    Generates AI-based proxies for missing critical columns using:
    1. Cross-column correlations (Rules).
    2. Statistical defaults (Fallbacks).
    
    Provides 'Transparency Metadata' to explain where data came from.
    """

    @staticmethod
    def generate_proxies(df: pd.DataFrame, mapping: dict, sector: str) -> tuple[pd.DataFrame, dict]:
        """
        Enriches the dataframe with proxy columns for any missing concepts.
        
        Args:
            df: Original DataFrame.
            mapping: Output from SemanticMapper (dict of concepts).
            sector: Selected sector string.
            
        Returns:
            (enriched_df, updated_mapping)
        """
        df = df.copy()
        updated_mapping = mapping.copy()
        
        # Helper to get mapped col
        def get_col(concept):
            entry = updated_mapping.get(concept)
            return entry['column'] if entry else None

        # --- Rule-Based Generation ---
        
        # Rule 1: Missing DROPOUT in Education
        if sector == "Education" and updated_mapping.get("DROPOUT") is None:
            # Check for LITERACY (Simulated check, assuming we map Literacy if it exists)
            # For this MVP, let's scan for a 'literacy' column strictly if not in map
            lit_col = None
            for col in df.columns:
                if 'literacy' in col.lower():
                    lit_col = col
                    break
            
            if lit_col:
                # Proxy: Dropout = 1 - Normalized(Literacy)
                # Assume Literacy is 0-100 or 0-1. Detect max.
                max_val = df[lit_col].max()
                scale = 100 if max_val > 1 else 1
                
                # Generating Proxy
                proxy_name = f"proxy_DROPOUT"
                df[proxy_name] = (scale - df[lit_col]) / scale
                df[proxy_name] = df[proxy_name].clip(0, 1) # Ensure bounds
                
                updated_mapping["DROPOUT"] = {
                    'column': proxy_name,
                    'confidence': 0.7, # Medium confidence
                    'is_proxy': True,
                    'source': f"Inverted from {lit_col}"
                }
            else:
                # Fallback: Statistical Default
                # Generate random noise around a 'reasonable' mean? 
                # Or just flat mean.
                # User Transparency: "Estimated using Statistical Default (Low Confidence)"
                proxy_name = f"proxy_DROPOUT_stat"
                df[proxy_name] = 0.15 + np.random.normal(0, 0.02, size=len(df)) # Mean 15%
                df[proxy_name] = df[proxy_name].clip(0, 1)
                
                updated_mapping["DROPOUT"] = {
                    'column': proxy_name,
                    'confidence': 0.3, # Low
                    'is_proxy': True,
                    'source': "Sector Statistical Default"
                }

        # Rule 2: Missing POVERTY (Generic)
        if updated_mapping.get("POVERTY") is None:
             proxy_name = "proxy_POVERTY"
             # Just use a placeholder distribution for MVP demo
             df[proxy_name] = 0.3 + np.random.normal(0, 0.05, size=len(df))
             df[proxy_name] = df[proxy_name].clip(0, 1)
             
             updated_mapping["POVERTY"] = {
                'column': proxy_name,
                'confidence': 0.2,
                'is_proxy': True,
                'source': "Statistical Estimation (Low Confidence)"
             }

        # Ensure all required concepts have *something*
        # (This handles any other missing concept using a generic fallback)
        for concept, meta in updated_mapping.items():
            if meta is None or (isinstance(meta, dict) and 'column' not in meta): # None or empty
                # Generic fallback
               col_name = f"est_{concept.lower()}"
               df[col_name] = 0.5 # Neutral value
               updated_mapping[concept] = {
                   'column': col_name,
                   'confidence': 0.1,
                   'is_proxy': True,
                   'source': "System Default (Missing Data)"
               }

        return df, updated_mapping
