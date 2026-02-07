import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

from src.semantic_mapper import SemanticMapper
from src.proxy_generator import ProxyGenerator
from src.need_engine import NeedEngine

def test_intelligent_engine():
    print("Testing Intelligent Sector Engine Logic...")
    
    # 1. Setup Mock Data (Obscure names to test inference)
    df = pd.DataFrame({
        'district_name': ['A', 'B', 'C'],
        'student_count': [100, 200, 300],
        'education_budget': [1000, 2000, 3000],
        'bpl_percentage': [0.1, 0.4, 0.8], # Poverty Proxy
        'literacy_rate': [0.9, 0.7, 0.5]  # Used to deduce Dropout
        # Missing 'DROPOUT' explicitly
    })
    
    sector = "Education"
    print(f"\n[Input Data - Sector: {sector}]")
    print(df.columns.tolist())
    
    # 2. Test Auto-Mapping
    print("\n[Testing Semantic Mapper]")
    mapping = SemanticMapper.infer_mapping(df, sector)
    
    for concept, meta in mapping.items():
        if meta:
            print(f"  {concept} -> {meta['column']} (Conf: {meta['confidence']})")
        else:
            print(f"  {concept} -> MISSING")
            
    # Expectation: 
    # REGION -> district_name
    # POPULATION -> student_count
    # AID -> education_budget
    # POVERTY -> bpl_percentage
    # DROPOUT -> MISSING
    
    # 3. Test Proxy Generation
    print("\n[Testing Proxy Generator]")
    enriched_df, full_mapping = ProxyGenerator.generate_proxies(df, mapping, sector)
    
    dropout_meta = full_mapping.get('DROPOUT')
    print(f"  DROPOUT Proxy: {dropout_meta}")
    
    if dropout_meta and dropout_meta['is_proxy']:
        print("[SUCCESS] DROPOUT Proxy Generated successfully!")
        print(f"   Source: {dropout_meta['source']}")
        print(enriched_df[['district_name', 'literacy_rate', dropout_meta['column']]])
    else:
        print("[FAILURE] Failed to generate DROPOUT proxy.")
        
    # 4. Test Need Calculation with Proxies
    print("\n[Testing Need Engine with Proxies]")
    need_df = NeedEngine.calculate_need(enriched_df, full_mapping, sector)
    print(need_df[['district_name', 'need_score']])
    
    if 'need_score' in need_df.columns:
        print("[SUCCESS] Need Score calculated successfully using mapped columns.")

if __name__ == "__main__":
    test_intelligent_engine()
