import streamlit as st
import pandas as pd
import time
import requests
import json

# API Base URL
API_URL = "http://localhost:8000/api"

# Page Config
st.set_page_config(page_title="Secure Federated Healthcare", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a clean modern look
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-border {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .dark-mode-override {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Secure Federated Learning for Healthcare")
st.markdown("**Privacy-Preserving Disease Risk Prediction Framework**")
st.markdown("This prototype simulates multiple healthcare institutions collaboratively training a predictive AI model without ever centralizing raw patient data. It now uses a decoupled **Web Backend Orchestrator (FastAPI)**.")

# Sidebar Configuration
st.sidebar.header("🌐 Network Configuration")
num_clients = st.sidebar.slider("Number of Hospitals", min_value=2, max_value=10, value=3)
num_rounds = st.sidebar.slider("Training Rounds", min_value=1, max_value=30, value=10)

st.sidebar.header("🔒 Privacy Configuration")
privacy_level = st.sidebar.selectbox("Privacy Level (DP Noise)", ["Low", "Medium", "High", "Extreme"], help="Higher privacy adds more noise to the aggregated model weights.")

privacy_map = {
    "Low": 0.0,
    "Medium": 0.5,
    "High": 1.5,
    "Extreme": 3.0
}
privacy_multiplier = privacy_map[privacy_level]

st.sidebar.markdown("---")
start_btn = st.sidebar.button("🚀 Start Collaborative Training via API", use_container_width=True)
st.sidebar.info("The Frontend is now entirely detached! Clicking this signals the FL Backend to boot the federated mesh securely.")

if start_btn:
    st.session_state['training_active'] = True
    try:
        # Trigger the backend
        res = requests.post(f"{API_URL}/start_training", json={
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "privacy_multiplier": privacy_multiplier
        })
        if res.status_code != 200:
            st.error("Failed to connect to the FL Backend.")
            st.session_state['training_active'] = False
    except Exception as e:
        st.error(f"Error communicating with backend: {e}")
        st.session_state['training_active'] = False

# Dashboard Layout
st.markdown("### 📊 Live Backend Metrics")
col1, col2, col3, col4 = st.columns(4)
round_metric = col1.empty()
acc_metric = col2.empty()
leakage_metric = col3.empty()
status_metric = col4.empty()

round_metric.metric("Current Round", "0 / 0")
acc_metric.metric("Global Accuracy", "0.0%")
leakage_metric.metric("Data Leakage Risk", "0.0%")
status_metric.metric("Backend Status", "Idle")

st.markdown("### 📈 Accuracy vs. Privacy Risk Trade-off")
chart_col = st.empty()

if st.session_state.get('training_active', False):
    progress_bar = st.progress(0)
    
    while True:
        try:
            status_res = requests.get(f"{API_URL}/status")
            status_data = status_res.json()
            is_training = status_data.get("status") == "training"
            
            if is_training:
                status_metric.metric("Backend Status", "Training...", delta="Active")
            else:
                st.session_state['training_active'] = False
                status_metric.metric("Backend Status", "Completed ✓")
                progress_bar.progress(1.0)
                
            metrics_res = requests.get(f"{API_URL}/metrics")
            data = metrics_res.json().get("metrics", [])
            
            if len(data) > 0:
                df = pd.DataFrame(data)
                latest_round = df.iloc[-1]['round']
                global_acc = df.iloc[-1].get('global_accuracy', 0.0)
                leakage_risk = df.iloc[-1]['leakage_risk']
                
                round_metric.metric("Current Round", f"{int(latest_round)} / {num_rounds}")
                
                if len(df) > 1 and 'global_accuracy' in df.columns:
                    prev_acc = df.iloc[-2].get('global_accuracy', 0.0)
                    acc_delta = f"{(global_acc - prev_acc)*100:.2f}%"
                else:
                    acc_delta = None
                    
                acc_metric.metric("Global Accuracy", f"{global_acc*100:.2f}%", delta=acc_delta)
                leakage_metric.metric("Data Leakage Risk", f"{leakage_risk*100:.1f}%")
                
                progress_bar.progress(min(int(latest_round) / num_rounds, 1.0))
                
                if 'global_accuracy' in df.columns:
                    chart_data = df[['round', 'global_accuracy', 'leakage_risk']].set_index('round')
                    chart_data.columns = ['Global Accuracy (0-1)', 'Leakage Risk (0-1)']
                    chart_col.line_chart(chart_data)
                    
            if not is_training:
                break
                
        except Exception as e:
            st.error(f"Backend Server offline? Error: {e}")
            break
            
        time.sleep(1)
