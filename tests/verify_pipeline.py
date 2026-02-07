import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.profiler import ColumnProfiler
from src.inference import SemanticInferenceEngine
from src.normalization import NormalizationLayer
from src.imputation import DerivedIndicatorGenerator
from src.need_engine import NeedEngine
from src.fairness import FairnessEngine
from src.reallocation import ReallocationEngine

def test_pipeline():
    print("--- Starting Pipeline Verification ---")
    
    # 1. Create Mock Data
    data = {
        'District': ['Region A', 'Region B', 'Region C', 'Region D'],
        'Total_Population': [1000, 2000, 1500, 5000],
        'Education_Funds': [50000, 10000, 120000, 40000], # A=50/head, B=5/head, C=80/head, D=8/head
        'Literacy_Rate': [90, 40, 95, 50],
        'Poverty_Rate': [10, 80, 5, 60]
    }
    df = pd.DataFrame(data)
    print("Masked Data Created:")
    print(df)
    
    # 2. Profile
    profile = ColumnProfiler.profile(df)
    print("\nProfiling Complete.")
    
    # 3. Inference
    inferred = SemanticInferenceEngine.infer_roles(profile)
    print("\nInference Result:")
    for role, info in inferred.items():
        print(f"  {role}: {info}")
        
    # Verify expected inference
    assert inferred['REGION']['column'] == 'District'
    assert inferred['POPULATION']['column'] == 'Total_Population'
    assert inferred['AID']['column'] == 'Education_Funds'
    assert inferred['OUTCOME']['column'] == 'Literacy_Rate'
    assert inferred['POVERTY']['column'] == 'Poverty_Rate'
    
    # 4. Normalization
    mapping = {k: v for k, v in inferred.items()} # Use inferred directly
    norm_df = NormalizationLayer.process(df, mapping)
    print("\nNormalization Complete.")
    print(norm_df[['norm_region', 'norm_aid', 'norm_outcome']].head(2))
    
    # 5. Imputation (Derived Indicators)
    imp_df = DerivedIndicatorGenerator.derive_indicators(norm_df, mapping)
    print("\nImputation Complete. Check derived 'dropout' (1-Literacy):")
    # outcome is literacy (0-1), dropout should be derived as 1-outcome
    if 'norm_dropout' in imp_df.columns:
        print(imp_df[['norm_outcome', 'norm_dropout']])
    
    # 6. Need Calculation (Education)
    imp_df = NeedEngine.calculate_need(imp_df, mapping, "Education")
    print("\nNeed Calculation:")
    print(imp_df[['norm_region', 'need_score']])
    
    # 7. Fairness
    fair_df = FairnessEngine.calculate_fairness(imp_df, 
                                               aid_col='norm_aid', 
                                               population_col='norm_population', 
                                               need_score_col='need_score')
    print("\nFairness Scores:")
    print(fair_df[['norm_region', 'fairness_score']])
    
    # 8. Reallocation
    # Region B and D have low funding and high poverty/low literacy -> Should have high Need.
    # Region A and C have high funding and low need.
    # Expect B or D to be Receivers. C to be Donor.
    
    realloc_df = ReallocationEngine.simulate(fair_df, 
                                            aid_col='norm_aid', 
                                            pop_col='norm_population', 
                                            need_col='need_score')
    print("\nSimulation Result:")
    print(realloc_df[['norm_region', 'norm_aid', 'new_aid', 'transfer_amount', 'role']])
    
    print("\n--- Pipeline Verified Successfully ---")

if __name__ == "__main__":
    test_pipeline()
