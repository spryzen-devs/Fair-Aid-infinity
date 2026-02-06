import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

import importlib

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force reload of local modules to pick up changes without restarting Streamlit
import src.data_loader
import src.need_engine
import src.fairness
import src.reallocation
import src.semantic_mapper
import src.proxy_generator

importlib.reload(src.data_loader)
importlib.reload(src.need_engine)
importlib.reload(src.fairness)
importlib.reload(src.reallocation)
importlib.reload(src.semantic_mapper)
importlib.reload(src.proxy_generator)

from src.data_loader import DataLoader
from src.need_engine import NeedEngine
from src.fairness import FairnessEngine
from src.reallocation import ReallocationEngine
from src.semantic_mapper import SemanticMapper
from src.proxy_generator import ProxyGenerator

# --- Configuration ---
st.set_page_config(
    page_title="FairAid-Infinity System",
    page_icon="⚖️",
    layout="wide"
)

# --- Session State ---
if 'mvp_step' not in st.session_state:
    st.session_state['mvp_step'] = 1 # 1: Sector, 2: Upload, 3: Map, 4: Dashboard
if 'sector' not in st.session_state:
    st.session_state['sector'] = None
if 'raw_df' not in st.session_state:
    st.session_state['raw_df'] = None
if 'mapping' not in st.session_state:
    st.session_state['mapping'] = {}
if 'final_df' not in st.session_state:
    st.session_state['final_df'] = None
if 'simulation_run' not in st.session_state:
    st.session_state['simulation_run'] = False

# --- UI Header ---
st.title("FairAid-Infinity ⚖️")
st.markdown("### Intelligent Aid Allocation System")

# --- Step 1: Sector Selection ---
if st.session_state['mvp_step'] == 1:
    st.header("Step 1: Select Sector")
    sector = st.radio(
        "Choose the aid sector you want to analyze:",
        ["Education", "Health", "Food Security", "Disaster Relief"],
        index=None
    )
    
    if sector:
        if st.button("Confirm Sector"):
            st.session_state['sector'] = sector
            st.session_state['mvp_step'] = 2
            st.rerun()

# --- Step 2: Upload Data ---
elif st.session_state['mvp_step'] == 2:
    st.header(f"Step 2: Upload Data for {st.session_state['sector']}")
    st.info("Upload one or more CSV files. They will be merged into a single dataset.")
    
    uploaded_files = st.file_uploader("Upload CSVs", type=['csv'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process & Auto-Map"):
            with st.spinner("Profiling Data & Inferring Columns..."):
                dfs = DataLoader.load_files(uploaded_files)
                master_df = pd.concat(dfs.values(), ignore_index=True)
                st.session_state['raw_df'] = master_df
                
                # Run Auto-Mapping
                mn = SemanticMapper.infer_mapping(master_df, st.session_state['sector'])
                
                # Check for missing & Generate Proxies
                # We do this immediately to show the "Proposal" to the user
                enriched_df, full_mapping = ProxyGenerator.generate_proxies(
                    master_df, mn, st.session_state['sector']
                )
                
                st.session_state['raw_df'] = enriched_df
                st.session_state['mapping'] = full_mapping
                
                st.session_state['mvp_step'] = 3
                st.rerun()

# --- Step 3: Transparent Mapping Review ---
elif st.session_state['mvp_step'] == 3:
    st.header("Step 3: Data Analysis Report")
    st.markdown("We have automatically mapped your data and generated transparency reports for any AI-estimated values.")
    
    map_cfg = st.session_state['mapping']
    df = st.session_state['raw_df']
    
    # 1. Visualization of Mappings
    st.subheader("📊 Mapped Variables & Confidence")
    
    # Grid Layout
    cols = st.columns(3)
    
    # We iterate through the Schema requirement order for clarity
    schema = SemanticMapper.SCHEMAS.get(st.session_state['sector'])
    required_concepts = schema['required'] if schema else map_cfg.keys()
    
    for idx, concept in enumerate(required_concepts):
        meta = map_cfg.get(concept)
        if not meta: continue
        
        container = cols[idx % 3].container(border=True)
        container.markdown(f"**{concept}**")
        
        # Color Code: Green (High Confidence), Orange (Proxy/Low)
        is_proxy = meta.get('is_proxy', False)
        confidence = meta.get('confidence', 0)
        
        if is_proxy:
            status_color = "orange"
            icon = "🤖"
        elif confidence > 0.8:
            status_color = "green"
            icon = "✅"
        else:
            status_color = "gold"
            icon = "⚠️"
            
        container.markdown(f":{status_color}[{icon} **{meta['column']}**]")
        container.caption(f"Confidence: {int(confidence*100)}% | Source: {meta.get('source', 'Inferred from Name')}")
        
    st.divider()
    
    # 2. Allow Override (Advanced)
    with st.expander("📝 Modify Mappings (Advanced)"):
        st.caption("If an inference is incorrect, select the correct column.")
        with st.form("manual_override"):
            new_selection = {}
            for concept in required_concepts:
                current = map_cfg.get(concept, {}).get('column')
                options = list(df.columns)
                # Ensure current is in options if it exists
                idx = options.index(current) if current in options else 0
                
                new_selection[concept] = st.selectbox(f"{concept}", options, index=idx)
            
            if st.form_submit_button("Update Mapping"):
                # Update mapping with manual overrides
                # Note: We lose proxy metadata if user manually maps.
                for k, v in new_selection.items():
                    st.session_state['mapping'][k] = {'column': v, 'confidence': 1.0, 'source': 'User Manual Override', 'is_proxy': False}
                st.rerun()

    if st.button("Confirm Analysis & Proceed", type="primary"):
        st.session_state['mvp_step'] = 4
        st.rerun()

# --- Step 4 & 5: Dashboard & Simulation ---
elif st.session_state['mvp_step'] == 4:
    map_cfg = st.session_state['mapping']
    
    # Always compute base metrics for visualization
    base_df = st.session_state['raw_df'].copy()
    
    # 1. Calculate Need (Using Unified Engine)
    base_df = NeedEngine.calculate_need(base_df, map_cfg, st.session_state['sector'])
    
    # 2. Calculate Fairness (Before)
    # Get column names dynamically
    aid_c = map_cfg.get('AID')['column']
    pop_c = map_cfg.get('POPULATION')['column']
    
    base_df = FairnessEngine.calculate_fairness(
        base_df, 
        aid_col=aid_c, 
        population_col=pop_c, 
        need_score_col='need_score'
    )
    
    # Use base_df for the "Before" metrics
    df = base_df

    # --- Dashboard View ---
    st.header(f"Fairness Analysis: {st.session_state['sector']}")
    
    # Top Metrics
    c1, c2, c3 = st.columns(3)
    
    threshold = df['fairness_score'].quantile(0.25)
    left_out_count = df[df['fairness_score'] <= threshold].shape[0]
    
    c1.metric("Total Regions", len(df))
    c2.metric("Critical Regions (Bottom 25%)", left_out_count, delta_color="inverse")
    c3.metric("Avg Fairness Score", f"{df['fairness_score'].mean():.2f}")
    
    st.divider()
    
    # Chart: Fairness Before
    st.subheader("1. Current Fairness Landscape")
    
    region_c = map_cfg.get('REGION')['column']
    
    q25 = df['fairness_score'].quantile(0.25)
    q75 = df['fairness_score'].quantile(0.75)
    
    df['status'] = ['Critical' if x <= q25 else 'Overserved' if x >= q75 else 'Neutral' for x in df['fairness_score']]
    
    fig_before = px.bar(
        df.sort_values('fairness_score'),
        x=region_c,
        y='fairness_score',
        color='status',
        color_discrete_map={'Critical': 'red', 'Overserved': 'green', 'Neutral': 'lightgray'},
        title="Fairness Score by Region (Lower is Worse)",
        text='fairness_score'
    )
    fig_before.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_before, use_container_width=True)
    
    # --- Step 5: Simulation ---
    st.divider()
    st.header("2. Reallocation Simulation")
    
    if not st.session_state['simulation_run']:
        st.info("""
        **🔄 Reallocation Logic:**
        The system automatically identifies **Overserved Regions** and re-routes a small portion (5%) of their surplus to **Critical Regions**.
        """)
        
        if st.button("Run Reallocation Simulation 🚀", type="primary"):
            sim_df = ReallocationEngine.simulate(
                base_df, 
                aid_col=aid_c, 
                pop_col=pop_c, 
                need_col='need_score'
            )
            st.session_state['sim_df'] = sim_df
            st.session_state['simulation_run'] = True
            st.rerun()
            
    else:
        # Show Results
        if 'sim_df' in st.session_state:
            sim_df = st.session_state['sim_df']
        else:
            sim_df = base_df
            st.error("Simulation data missing. Please reset.")
        
        # Improvement Metrics
        old_mean = sim_df['fairness_score'].mean()
        new_mean = sim_df['new_fairness_score'].mean()
        delta_pct = ((new_mean - old_mean) / old_mean) * 100
        
        m1, m2 = st.columns(2)
        m1.metric("Fairness Improvement", f"+{delta_pct:.1f}%", delta=f"{new_mean:.2f} (New Avg)")
        
        transferred = sim_df[sim_df['transfer_amount'] > 0]['transfer_amount'].sum()
        m2.metric("Total Aid Reallocated", f"${transferred:,.2f}")
        
        # Comparison Chart
        st.subheader("Fairness: Before vs After")
        
        sim_df['abs_change'] = (sim_df['new_fairness_score'] - sim_df['fairness_score']).abs()
        top_affected = sim_df.sort_values('abs_change', ascending=False).head(15)
        
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            x=top_affected[region_c],
            y=top_affected['fairness_score'],
            name='Before',
            marker_color='lightgray'
        ))
        fig_compare.add_trace(go.Bar(
            x=top_affected[region_c],
            y=top_affected['new_fairness_score'],
            name='After',
            marker_color='blue'
        ))
        
        fig_compare.update_layout(
            title="Top 15 Most Impacted Regions",
            barmode='group',
            xaxis_title="Region",
            yaxis_title="Fairness Score"
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Table
        st.subheader("Reallocation Details")
        display_cols = [region_c, 'role', aid_c, 'new_aid', 'transfer_amount', 'fairness_score', 'new_fairness_score']
        st.dataframe(sim_df[display_cols].style.format({
            aid_c: "${:,.0f}",
            'new_aid': "${:,.0f}",
            'transfer_amount': "${:+,.0f}",
            'fairness_score': "{:.2f}",
            'new_fairness_score': "{:.2f}"
        }))
        
        if st.button("Reset Analysis"):
            st.session_state['mvp_step'] = 1
            st.session_state['simulation_run'] = False
            st.session_state['sector'] = None
            st.session_state['raw_df'] = None
            st.rerun()

# --- Sidebar Footer ---
try:
    with st.sidebar:
        st.markdown("---")
        st.caption("FairAid-Infinity System v2.0")
        if st.button("Hard Reset"):
            st.session_state.clear()
            st.rerun()
except:
    pass
