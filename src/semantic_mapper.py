import pandas as pd
import numpy as np
import re

class SemanticMapper:
    """
    Handles automatic, sector-specific column mapping based on:
    1. Keyword tokens in column names.
    2. Data types and ranges (heuristic).
    """
    
    # --- Sector Schemas ---
    SCHEMAS = {
        "Education": {
            "required": ["REGION", "POPULATION", "AID", "POVERTY", "DROPOUT"],
            "keywords": {
                "REGION": ["region", "state", "district", "city", "location", "geo"],
                "POPULATION": ["pop", "student", "enrollment", "count", "census"],
                "AID": ["aid", "budget", "fund", "amount", "allocation", "spend", "cost"],
                "POVERTY": ["poverty", "income", "poor", "econ", "wealth", "bpl"],
                "DROPOUT": ["dropout", "drop", "out_of_school", "leave", "attrition"]
            }
        },
        "Health": {
            "required": ["REGION", "POPULATION", "AID", "POVERTY", "DISEASE_BURDEN"],
            "keywords": {
                "REGION": ["region", "state", "district", "city", "location"],
                "POPULATION": ["pop", "people", "patient", "count"],
                "AID": ["aid", "budget", "fund", "health_spend", "expenditure"],
                "POVERTY": ["poverty", "income", "poor"],
                "DISEASE_BURDEN": ["disease", "burden", "illness", "sick", "morbidity", "mortality", "prevalence"]
            }
        },
        # Map "Food Security" and "Disaster Relief" to similar schemas for MVP
        "Food Security": {
            "required": ["REGION", "POPULATION", "AID", "POVERTY", "MALNUTRITION"],
            "keywords": {
                "REGION": ["region", "state", "district"],
                "POPULATION": ["pop", "household", "family"],
                "AID": ["aid", "budget", "food_subsidy", "ration"],
                "POVERTY": ["poverty", "income", "poor"],
                "MALNUTRITION": ["malnutrition", "nutrition", "hunger", "stunting", "underweight"]
            }
        },
        "Disaster Relief": {
             "required": ["REGION", "POPULATION", "AID", "POVERTY", "DAMAGE_SEVERITY"],
             "keywords": {
                "REGION": ["region", "state", "district"],
                "POPULATION": ["pop", "affected", "victim"],
                "AID": ["aid", "budget", "relief", "fund"],
                "POVERTY": ["poverty", "vulnerability"],
                "DAMAGE_SEVERITY": ["damage", "severity", "impact", "loss", "destroyed"]
             }
        }
    }

    @staticmethod
    def infer_mapping(df: pd.DataFrame, sector: str) -> dict:
        """
        Scans dataframe and returns the best column candidate for each concept 
        in the sector schema.
        
        Returns:
            dict: { CONCEPT: {'column': col_name, 'confidence': float} }
        """
        schema = SemanticMapper.SCHEMAS.get(sector, SemanticMapper.SCHEMAS["Education"])
        mapping = {}
        used_columns = set()
        
        # 1. Normalize Column Names for Matching
        # Create a dict of {clean_name: original_name}
        clean_cols = {}
        for col in df.columns:
            # lower, remove special chars
            clean = re.sub(r'[^a-zA-Z0-9]', '_', str(col).lower())
            clean_cols[col] = clean

        # 2. Match for each required concept
        for concept in schema['required']:
            best_col = None
            best_score = 0.0
            kw_list = schema['keywords'].get(concept, [])
            
            for col, clean_name in clean_cols.items():
                if col in used_columns:
                     continue
                
                score = 0.0
                
                # A. Token overlap
                for kw in kw_list:
                    if kw in clean_name:
                        score += 0.5
                        # Exact match bonus
                        if kw == clean_name:
                            score += 0.3
                
                # B. Data Type Heuristic (Simple)
                if concept == "REGION":
                    # Prefer Object/String
                    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                         score += 0.2
                else:
                    # Prefer Numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        score += 0.2
                    else:
                        # Heavy penalty if mapping numeric concept to text
                        score -= 0.5
                        
                if score > best_score:
                    best_score = score
                    best_col = col
            
            # 3. Assign
            if best_col and best_score > 0.3: # Threshold
                mapping[concept] = {
                    'column': best_col,
                    'confidence': min(best_score, 1.0) # Cap at 1.0
                }
                used_columns.add(best_col)
            else:
                # Mark as missing
                mapping[concept] = None
                
        return mapping
