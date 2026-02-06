from typing import Dict, Any, List, Tuple

class SemanticInferenceEngine:
    """
    Infers the conceptual role of columns based on metadata profiling.
    """
    
    ROLES = [
        "REGION", "POPULATION", "AID", "POVERTY", 
        "OUTCOME", "RISK", "DISASTER"
    ]
    
    # Keyword weights for scoring
    KEYWORDS = {
        "REGION": ["state", "district", "region", "area", "zone", "province", "geo"],
        "POPULATION": ["population", "pop", "people", "census", "inhabitants", "demographic"],
        "AID": ["aid", "fund", "budget", "allocation", "amount", "cost", "expenditure", "grant", "subsidy"],
        "POVERTY": ["poverty", "poor", "income", "bpl", "deprivation", "wealth", "gdp"],
        "OUTCOME": ["literacy", "score", "rate", "grad", "health", "expectancy", "mortality"],
        "RISK": ["risk", "vulnerability", "hazard", "exposure"],
        "DISASTER": ["disaster", "flood", "drought", "quake", "damage", "severity"]
    }
    
    @staticmethod
    def infer_roles(profile_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Returns a mapping of conceptual roles to suggested columns with confidence scores.
        Output format:
        {
            "REGION": {"column": "District_Name", "confidence": 0.95},
            "POPULATION": {"column": "Pop_2022", "confidence": 0.8},
            ...
        }
        """
        role_mapping = {}
        
        # We need to find the best column for each role
        for role in SemanticInferenceEngine.ROLES:
            best_col = None
            best_score = 0
            
            for col_name, meta in profile_data.items():
                score = SemanticInferenceEngine._calculate_confidence(role, col_name, meta)
                if score > best_score:
                    best_score = score
                    best_col = col_name
            
            # Threshold for suggestion (can be low, we just want the best guess)
            if best_col and best_score > 0.3:
                role_mapping[role] = {
                    "column": best_col,
                    "confidence": round(best_score, 2)
                }
                
        return role_mapping

    @staticmethod
    def _calculate_confidence(role: str, col_name: str, meta: Dict[str, Any]) -> float:
        score = 0.0
        tokens = meta.get('tokens', [])
        col_type = meta.get('type', '')
        
        # 1. Keyword Matching (Primary signal)
        keywords = SemanticInferenceEngine.KEYWORDS.get(role, [])
        for token in tokens:
            if token in keywords:
                score += 0.6  # Strong match
            elif any(k in token for k in keywords):
                score += 0.3  # Partial match
        
        # 2. Type Heuristics
        if role == "REGION":
            if col_type == 'categorical': # Regions are usually names
                score += 0.2
            elif col_type == 'numeric': # Regions are rarely numeric unless IDs
                score -= 0.3
        
        elif role in ["POPULATION", "AID", "POVERTY"]:
            if col_type == 'numeric':
                score += 0.2
            else:
                score -= 0.5 # These MUST be numeric
        
        # 3. Value Heuristics (Simple)
        if role == "POPULATION" and col_type == 'numeric':
            if meta.get('mean', 0) > 1000: # Populations are usually large
                score += 0.1
                
        if role == "OUTCOME" and col_type == 'numeric':
            # Rates are often 0-1 or 0-100
            input_max = meta.get('max', 0)
            if input_max <= 100:
                score += 0.1

        return min(1.0, max(0.0, score))
