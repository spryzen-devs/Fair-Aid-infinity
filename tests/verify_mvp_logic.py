import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

from src.need_engine import NeedEngine
from src.fairness import FairnessEngine
from src.reallocation import ReallocationEngine

def test_mvp_logic():
    print("Testing Winning MVP Logic...")
    
    # 1. Setup Mock Data
    df = pd.DataFrame({
        'region': ['A', 'B', 'C', 'D'],
        'pop': [100, 100, 100, 100],
        'aid': [1000, 200, 50, 2000],  # D is overserved, C is underserved
        'poverty': [0.5, 0.5, 0.8, 0.1], # C has high need
        'dropout': [0.1, 0.1, 0.2, 0.0]
    })
    
    print("\n[Input Data]")
    print(df)
    
    # 2. Test Need Engine
    print("\n[Testing Need Engine]")
    # We expect normalization 0-1. 
    # Poverty: 0.1->0.0, 0.5->0.57, 0.8->1.0 (approx)
    df = NeedEngine.calculate_need(df, ['poverty', 'dropout'])
    print(df[['region', 'need_score']])
    
    # 3. Test Fairness Engine
    print("\n[Testing Fairness Engine]")
    # Formula: (aid/pop) / (need + 0.01)
    df = FairnessEngine.calculate_fairness(df, 'aid', 'pop', 'need_score')
    print(df[['region', 'aid_per_capita', 'fairness_score']])
    
    # 4. Test Reallocation
    print("\n[Testing Reallocation Engine]")
    df = ReallocationEngine.simulate(df, 'aid', 'pop', 'need_score')
    
    print(df[['region', 'role', 'aid', 'new_aid', 'transfer_amount']])
    
    # Verify Conservation of Aid
    total_old = df['aid'].sum()
    total_new = df['new_aid'].sum()
    diff = abs(total_old - total_new)
    
    print(f"\nTotal Aid Before: {total_old}, After: {total_new}")
    if diff < 1.0:
        print("✅ Aid Conserved!")
    else:
        print("❌ Aid LEAKAGE detected!")
        
    # Verify Overserved was Taxed
    overserved = df[df['role'] == 'Overserved']
    if not overserved.empty:
        taxed_amount = overserved['transfer_amount'].sum()
        print(f"Overserved regions taxed total: {taxed_amount}")
        if taxed_amount < 0:
             print("✅ Overserved regions were taxed.")
        else:
             print("❌ Overserved regions were NOT taxed.")

    # Verify Underserved Received
    underserved = df[df['role'] == 'Underserved']
    if not underserved.empty:
        received = underserved['transfer_amount'].sum()
        print(f"Underserved regions received total: {received}")
        if received > 0:
             print("✅ Underserved regions received aid.")
        else:
             print("❌ Underserved regions received NOTHING.")

if __name__ == "__main__":
    test_mvp_logic()
