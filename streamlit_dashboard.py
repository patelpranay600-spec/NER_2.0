import streamlit as st
import geopandas as gpd
import pandas as pd
import networkx as nx
import xgboost as xgb
import folium
from streamlit_folium import st_folium
import os
import random

# --- Config & Paths ---



# --- Config & Paths ---
st.set_page_config(page_title="NER AI Logistics Intelligence", layout="wide")

import os
# Get the absolute path of the directory where this script lives (scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the 'app' directory, then into 'data/processed'
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")

ROADS_GPKG = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
REAL_TARGET_DATA = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")
MODEL_FILE = os.path.join(PROCESSED_DIR, "xgboost_production_model.json")

# --- Helper Functions ---
@st.cache_resource
def load_data_and_model():
    roads = gpd.read_file(ROADS_GPKG)
    
    # ADD THIS LINE: Force standard GPS coordinates for web mapping
    roads = roads.to_crs("EPSG:4326")
    
    df = pd.read_parquet(REAL_TARGET_DATA)
    # ... (rest of the function stays the same)
    
    # Get latest date for live simulation
    latest_date = df['valid_time'].max()
    df_current = df[df['valid_time'] == latest_date].copy()
    
    # Load ML Model
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_FILE)
    
    features = [
        'length_km_calculated', 'slope_mean', 'slope_max', 'roughness_mean', 
        'twi_mean', 'twi_max', 'log_flow_accum_mean', 'log_flow_accum_max',
        'rainfall_24h', 'rainfall_3d', 'rainfall_7d', 'rainfall_anomaly',
        'ndvi', 'mndwi', 's1_vv', 's1_vh', 'vh_vv_ratio', 'pred_label' # <-- Add this!
    ]
    
    # Generate Logistics Vulnerability Index (LVI)
    df_current['LVI_Score'] = clf.predict_proba(df_current[features].fillna(0.0))[:, 1]
    
    roads['road_id'] = roads['road_id'].astype(str)
    df_current['road_id'] = df_current['road_id'].astype(str)
    roads_risk = pd.merge(roads, df_current[['road_id', 'LVI_Score']], on='road_id', how='left')
    roads_risk['LVI_Score'] = roads_risk['LVI_Score'].fillna(0.0)
    
    return roads_risk, latest_date

@st.cache_resource
def build_network(_roads_risk):
    G = nx.Graph()
    alpha = 30 # High penalty for risky roads
    
    for idx, row in _roads_risk.iterrows():
        geom = row.geometry
        
        # FIX: Handle both standard LineStrings and MultiLineStrings
        if geom.geom_type == 'LineString':
            lines = [geom]
        elif geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        else:
            continue # Skip any weird geometry types
            
        dist = row['length_km_calculated']
        risk = row['LVI_Score']
        
        # Risk-penalized cost logic
        dynamic_cost = dist * (1 + (alpha * (risk ** 3)))
        
        # Add edges for all line segments
        for line in lines:
            coords = list(line.coords)
            start_node, end_node = coords[0], coords[-1]
            
            G.add_edge(start_node, end_node, road_id=row['road_id'], 
                       normal_weight=dist, dynamic_weight=dynamic_cost, risk=risk)
                   
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc), list(G.subgraph(largest_cc).nodes())
# --- UI Layout ---
st.title("🚛 NER Smart Logistics & Accessibility Platform")
st.markdown("**AI-Powered Route Prediction, Flood/Landslide Warning, & Infrastructure Monitoring**")

with st.spinner("Initializing AI Engine & GIS Data..."):
    roads_risk, latest_date = load_data_and_model()
    G_sub, nodes = build_network(roads_risk)

# Sidebar Controls
st.sidebar.header("Logistics Control Panel")
st.sidebar.write(f"**Current AI Forecast Date:** {latest_date.date()}")

# Select Route
st.sidebar.subheader("Route Planning")
# For demo purposes, we randomly select nodes, but display them as "Warehouses"
source_node = st.sidebar.selectbox("Origin (Supply Hub)", options=nodes[:50], format_func=lambda x: f"Hub {hash(x) % 1000}")
target_node = st.sidebar.selectbox("Destination (Delivery)", options=nodes[-50:], format_func=lambda x: f"District {hash(x) % 1000}")

# Field Reporting Simulation
st.sidebar.subheader("📱 Field Officer Reporting")
report_type = st.sidebar.selectbox("Incident Type", ["None", "Landslide", "Road Washed Out (Flood)", "Bridge Collapsed"])
if report_type != "None":
    st.sidebar.error(f"🚨 IMMEDIATE ALERT: {report_type} reported. Graph dynamically updating...")
    # In a real app, this would force LVI_Score to 1.0 for the nearest road

# --- Routing Engine ---
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Routing Intelligence")
    try:
        # Standard vs Safe Route
        normal_path = nx.shortest_path(G_sub, source=source_node, target=target_node, weight='normal_weight')
        safe_path = nx.shortest_path(G_sub, source=source_node, target=target_node, weight='dynamic_weight')
        
        normal_dist = nx.path_weight(G_sub, normal_path, 'normal_weight')
        safe_dist = nx.path_weight(G_sub, safe_path, 'normal_weight')
        
        # Calculate max risk encountered on normal route
        normal_risk = max([G_sub[u][v]['risk'] for u, v in zip(normal_path[:-1], normal_path[1:])])
        
        st.info(f"**Standard GPS Route:**\n\nDistance: {normal_dist:.2f} km\n\nEst. Time: {(normal_dist/30)*60:.0f} mins")
        
        if normal_risk > 0.4:
            st.error(f"⚠️ **CRITICAL VULNERABILITY DETECTED**\n\nThe standard route crosses a segment with a {(normal_risk*100):.1f}% AI predicted failure rate (Flood/Landslide).")
            st.success(f"✅ **AI Alternative Route Generated:**\n\nDistance: {safe_dist:.2f} km\n\nEst. Time: {(safe_dist/30)*60:.0f} mins\n\n*Safely bypasses high-risk infrastructure.*")
        else:
            st.success("✅ Standard route is currently clear of weather/terrain disruptions.")
            
    except Exception as e:
        st.warning("Could not calculate route between these specific points. Try selecting different hubs.")

# --- Map Rendering ---
# --- Map Rendering ---
# --- Map Rendering ---
with col1:
    st.subheader("Live Vulnerability Map")
    
    try:
        bounds = roads_risk.total_bounds
        center_lat, center_lon = (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2
        
        # FIX: Changed tiles to OpenStreetMap (100% Free, no API Key needed)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="OpenStreetMap")
        
        high_risk = roads_risk[roads_risk['LVI_Score'] > 0.4].nlargest(300, 'LVI_Score')
        
        for _, row in high_risk.iterrows():
# ... (rest of the drawing logic stays the same)
            geom = row.geometry
            if geom.geom_type == 'LineString':
                lines = [geom]
            elif geom.geom_type == 'MultiLineString':
                lines = list(geom.geoms)
            else:
                continue
                
            for line in lines:
                coords = [(y, x) for x, y in line.coords]
                folium.PolyLine(coords, color="red", weight=3, opacity=0.8).add_to(m)
            
        # Draw the Safe Route
        if 'safe_path' in locals() and len(safe_path) > 1:
            route_coords = [(node[1], node[0]) for node in safe_path]
            folium.PolyLine(route_coords, color="#00FF00", weight=5, opacity=1.0, tooltip="AI Safe Route").add_to(m)

        # OPTIMIZATION: returned_objects=[] stops the map from causing memory overflow lag
        st_folium(m, width=800, height=500, returned_objects=[])
        
    except Exception as e:
        st.error(f"Map Rendering Error: {e}")