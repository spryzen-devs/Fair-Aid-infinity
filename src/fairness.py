import pandas as pd
import numpy as np

class FairnessEngine:
    """
    Calculates fairness scores based on the 'Winning MVP' formula:
    fairness = (aid / population) / (need_score + epsilon)
    """

    @staticmethod
    def calculate_fairness(df: pd.DataFrame, 
                         aid_col: str, 
                         population_col: str, 
                         need_score_col: str) -> pd.DataFrame:
        """
        Computes fairness score for each region.
        
        Args:
            df: DataFrame containing the data.
            aid_col: Column name for quantitative aid (e.g. Budget).
            population_col: Column name for population (e.g. Student Count).
            need_score_col: Column name for aggregated need score (0-1 typically).
            
        Returns:
            DataFrame with 'fairness_score' and 'aid_per_capita' added.
        """
        # Avoid side effects
        df = df.copy()
        
        # 1. Aid per capita
        # Handle zero population to avoid infinity
        df['aid_per_capita'] = df[aid_col] / df[population_col].replace(0, 1)
        
        # 2. Fairness Score
        # Formula: aid_per_capita / (need + epsilon)
        # Interpretation: High Score = Good (Overserved), Low Score = Bad (Underserved)
        epsilon = 0.01
        
        # Ensure need score acts as a denominator correctly. 
        # If need is 0, score explodes (overserved).
        df['fairness_score'] = df['aid_per_capita'] / (df[need_score_col] + epsilon)
        
        return df
