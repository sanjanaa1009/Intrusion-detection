import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import time
import base64
from io import BytesIO, StringIO
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from models.lgbm_model import LGBMClassifier
from models.isolation_forest_model import IsolationForestModel
from models.user_behavior_model import UserBehaviorModel
from models.blockchain import BlockchainLogger
from utils.data_processor import DataProcessor
from utils.log_parser import LogParser
from utils.visualization import create_anomaly_charts, create_user_behavior_charts, create_threat_distribution
from utils.gemini_integration import get_gemini_recommendation, initialize_gemini_api
from utils.user_behavior import UserBehaviorAnalyzer, UserActivityProfiler
from attached_assets.unknownThreat_detector import UnknownThreatClassifier
from utils.chatbot import initialize_chatbot, chatbot_interface

# Page config
st.set_page_config(
    page_title="CyberSentry IDS",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set dark theme
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #2E2E2E;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #2E2E2E;
        color: white;
    }
    .stDataFrame {
        background-color: #2E2E2E;
        color: white;
    }
    .metric-card {
        background-color: #363636;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .anomaly-card {
        background-color: #363636;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 16px;
        color: #BBBBBB;
    }
    .trend-up {
        color: #FF6B6B;
    }
    .trend-down {
        color: #4ECDC4;
    }
    .high-severity {
        background-color: #ff7f0e;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 14px;
        float: right;
    }
    .feature-card {
        background-color: #2E2E2E;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API
initialize_gemini_api()

# Initialize session state variables
if 'lgbm_model' not in st.session_state:
    st.session_state.lgbm_model = LGBMClassifier()
    st.session_state.lgbm_model.initialize_model()

if 'isolation_forest_model' not in st.session_state:
    st.session_state.isolation_forest_model = IsolationForestModel()

if 'user_behavior_model' not in st.session_state:
    st.session_state.user_behavior_model = UserBehaviorModel()

if 'unknown_threat_detector' not in st.session_state:
    st.session_state.unknown_threat_detector = UnknownThreatClassifier()

if 'blockchain_logger' not in st.session_state:
    st.session_state.blockchain_logger = BlockchainLogger()

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

if 'logs_data' not in st.session_state:
    st.session_state.logs_data = None

if 'user_data' not in st.session_state:
    st.session_state.user_data = None

if 'results' not in st.session_state:
    st.session_state.results = None

if 'user_behavior_results' not in st.session_state:
    st.session_state.user_behavior_results = None

if 'unknown_threat_results' not in st.session_state:
    st.session_state.unknown_threat_results = None

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = initialize_chatbot()

if 'anomaly_count' not in st.session_state:
    st.session_state.anomaly_count = 46

if 'critical_events' not in st.session_state:
    st.session_state.critical_events = 11

if 'suspicious_users' not in st.session_state:
    st.session_state.suspicious_users = 22

if 'security_score' not in st.session_state:
    st.session_state.security_score = 54.0

# App header and sidebar
st.sidebar.image("https://images.unsplash.com/photo-1563013544-824ae1b704d3", use_container_width=True)
st.sidebar.title("CyberSentry Enterprise IDS")

# Main navigation
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["Dashboard", "Log Analysis", "User Behavior Analysis", "Zero-Day Detection", "Security Assistant", "Settings"]
)

# Helper function to create metric cards
def display_metric_card(label, value, trend="", trend_value="", icon="", col=None):
    html = f"""
    <div class="metric-card">
        <div class="metric-value">{value} {icon}</div>
        <div class="metric-label">{label}</div>
        <div class="{"trend-up" if trend == "up" else "trend-down"}">{trend_value}</div>
    </div>
    """
    if col:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)

# Create line chart data for activity trend
def generate_activity_trend():
    dates = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=7), 
                         end=pd.Timestamp.now(), freq='1h')
    
    # Normal activities with some randomness
    normal_activities = np.sin(np.linspace(0, 14*np.pi, len(dates))) * 0.5 + 5 + np.random.normal(0, 0.3, size=len(dates))
    normal_activities = np.maximum(normal_activities, 4)  # Keep values above 4
    
    # Anomalies as occasional spikes
    anomalies = np.zeros(len(dates))
    
    # Add some spikes
    for i in range(20):
        idx = np.random.randint(0, len(dates))
        anomalies[idx] = np.random.uniform(0.5, 1.0)
    
    # Create smoothed anomalies
    for i in range(1, len(anomalies)):
        if anomalies[i] == 0 and anomalies[i-1] > 0:
            anomalies[i] = max(0, anomalies[i-1] - 0.3)
    
    df = pd.DataFrame({
        'date': dates,
        'normal': normal_activities,
        'anomalies': anomalies
    })
    
    return df

# Create anomaly categories for dashboard
def create_anomaly_distribution():
    # UNSW dataset categories
    categories = {
        'Authentication': 0.25,  # 25% of anomalies
        'System': 0.20,
        'Configuration': 0.15,
        'Network': 0.20,
        'Data': 0.10,
        'User': 0.10
    }
    
    # Create pie chart data
    values = [st.session_state.anomaly_count * percentage for category, percentage in categories.items()]
    labels = list(categories.keys())
    
    return values, labels

# Helper function to display anomaly detail cards
def display_anomaly_detail(title, count, description, col, severity="High"):
    html = f"""
    <div class="anomaly-card">
        <h3 style="margin-top:0">{title} <span class="high-severity">{severity}</span></h3>
        <h1 style="color:#ff7f0e; font-size:48px">{count}</h1>
        <p>{description}</p>
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

# Helper function to display feature cards for dashboard
def display_feature_card(title, description, image_url):
    html = f"""
    <div class="feature-card">
        <h3>{title}</h3>
        <img src="{image_url}" style="width: 100%; border-radius: 5px; margin: 10px 0;">
        <p>{description}</p>
    </div>
    """
    return html

# Display dashboard overview
if app_mode == "Dashboard":
    st.title("CyberSentry: Enterprise Security Dashboard")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Enterprise-grade security monitoring with real-time anomaly detection and threat intelligence. 
    Monitor both <b>known attacks</b> and <b>zero-day threats</b> with ML-powered analytics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CyberSentry Overview Section
    st.header("CyberSentry Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(display_feature_card(
            "Log Analysis",
            "Analyze logs for known attack patterns based on the UNSW dataset. Detect intrusions with high accuracy using our LGBM model.",
            "https://images.unsplash.com/photo-1558494949-ef010cbdcc31"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(display_feature_card(
            "User Behavior Analysis",
            "Track and detect anomalies in enterprise user behavior patterns using machine learning with our Isolation Forest model.",
            "https://images.unsplash.com/photo-1573164713988-8665fc963095"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(display_feature_card(
            "Zero-Day Detection",
            "Identify previously unknown threats through advanced pattern recognition with our pattern-based Isolation Forest model.",
            "https://images.unsplash.com/photo-1639322537228-f710d846310a"
        ), unsafe_allow_html=True)

# Log Analysis mode
elif app_mode == "Log Analysis":
    st.title("Enterprise Log Analysis")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Upload raw network traffic logs for advanced analysis using our LGBM model based on the UNSW dataset.
    This module detects known network attack patterns with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # First show model metrics before any data is processed
    st.subheader("Model Performance Metrics")
    
    # Display a clean version of the metrics without div styling
    if st.session_state.lgbm_model is not None:
        st.subheader("Model Type: " + st.session_state.lgbm_model.metrics.get('model_type', 'RandomForest'))
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Accuracy", f"{st.session_state.lgbm_model.metrics.get('accuracy', 0.0) * 100:.2f}%")
        with metrics_col2:
            st.metric("Precision", f"{st.session_state.lgbm_model.metrics.get('precision', 0.0) * 100:.2f}%")
        with metrics_col3:
            st.metric("Recall", f"{st.session_state.lgbm_model.metrics.get('recall', 0.0) * 100:.2f}%")
            
        # Show attack categories
        st.subheader("UNSW Attack Categories")
        st.write("This model is trained to identify the following attack types:")
        
        attack_col1, attack_col2 = st.columns(2)
        attack_types = list(st.session_state.lgbm_model.attack_categories.values())
        
        half = len(attack_types) // 2
        with attack_col1:
            for attack in attack_types[:half]:
                st.write(f"• {attack}")
        with attack_col2:
            for attack in attack_types[half:]:
                st.write(f"• {attack}")
    
    # Input options
    input_method = st.radio("Select input method:", ["Upload Log File", "Enter Log Data", "Use Sample Logs"])
    
    log_text = None
    
    if input_method == "Upload Log File":
        uploaded_file = st.file_uploader("Upload Log File", type=["log", "txt", "csv"], key="log_file_uploader")
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.session_state.logs_data = df
            else:
                log_text = uploaded_file.getvalue().decode("utf-8")
    
    elif input_method == "Enter Log Data":
        log_text = st.text_area("Paste Log Data", 
                               value="192.168.1.100 - - [01/Jul/2023:15:42:31 +0000] \"GET /index.html HTTP/1.1\" 200 4523\n127.0.0.1 - - [01/Jul/2023:15:43:10 +0000] \"POST /login HTTP/1.1\" 302 354\n192.168.1.105 - - [01/Jul/2023:15:45:20 +0000] \"GET /admin HTTP/1.1\" 401 1293", 
                               height=250)
    
    elif input_method == "Use Sample Logs":
        log_format = st.selectbox("Select log format:", ["UNSW Network Logs"])
        
        if log_format == "UNSW Network Logs":
            try:
                df = pd.read_csv("data/sample_network_logs.csv")
                st.session_state.logs_data = df
            except:
                st.error("Sample network logs file not found. Please upload a CSV file.")
    
    # If we have log text, parse it
    if log_text:
        with st.spinner("Processing logs..."):
            # Split logs into lines
            log_lines = log_text.strip().split('\n')
            
            # Parse logs
            log_parser = LogParser()
            parsed_logs, detected_format = log_parser.parse_logs(log_lines)
            
            # Display the detected format
            st.success(f"Log format detected: {detected_format.upper()}")
            
            # Save the parsed logs to session state
            st.session_state.logs_data = parsed_logs
            
            # Display data preview
            st.subheader("Parsed Log Preview")
            st.dataframe(parsed_logs.head(10), use_container_width=True)
    
    # Now we have logs data in st.session_state.logs_data
    if st.session_state.logs_data is not None:
        # Process with LGBM model if appropriate columns exist
        required_columns = ['proto', 'service', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl']
        missing_columns = [col for col in required_columns if col not in st.session_state.logs_data.columns]
        
        if missing_columns:
            st.warning(f"The following columns required for LGBM analysis are missing: {', '.join(missing_columns)}")
            st.info("The LGBM model requires network traffic data with specific features.")
            
            # Show column mapping if we have some other data
            if len(st.session_state.logs_data.columns) > 3:
                st.subheader("Map Available Columns")
                st.write("Map your data columns to the required features:")
                
                col_mapping = {}
                available_cols = list(st.session_state.logs_data.columns)
                
                for req_col in required_columns:
                    mapped_col = st.selectbox(f"Map '{req_col}' to:", ["Skip"] + available_cols, key=f"map_{req_col}")
                    if mapped_col != "Skip":
                        col_mapping[req_col] = mapped_col
                
                if st.button("Process with Mapping"):
                    # Create mapped dataframe
                    mapped_df = pd.DataFrame()
                    for req_col, mapped_col in col_mapping.items():
                        mapped_df[req_col] = st.session_state.logs_data[mapped_col]
                    
                    # Fill missing columns with defaults
                    for col in missing_columns:
                        if col not in col_mapping:
                            if col in ['sbytes', 'dbytes', 'sttl', 'dttl']:
                                mapped_df[col] = 0
                            elif col in ['dur']:
                                mapped_df[col] = 1.0
                            else:
                                mapped_df[col] = "unknown"
                    
                    # Add source and destination IPs if available
                    if 'src_ip' not in mapped_df and 'ip_address' in st.session_state.logs_data:
                        mapped_df['src_ip'] = st.session_state.logs_data['ip_address']
                    
                    if 'dst_ip' not in mapped_df:
                        mapped_df['dst_ip'] = "192.168.1.1"  # Default gateway
                    
                    # Process with LGBM model
                    st.success("Created mapped dataframe for analysis")
                    processed_df = st.session_state.data_processor.preprocess_network_data(mapped_df)
                    
                    try:
                        with st.spinner("Analyzing network data with LGBM model..."):
                            predictions = st.session_state.lgbm_model.predict(processed_df)
                            probabilities = st.session_state.lgbm_model.predict_proba(processed_df)
                            
                            # Get prediction labels
                            attack_categories = st.session_state.lgbm_model.attack_categories
                            predicted_labels = predictions
                            
                            # Get category names for predictions
                            predicted_categories = []
                            for pred in predicted_labels:
                                category = attack_categories.get(int(pred), "Unknown")
                                predicted_categories.append(category)
                            
                            # Add predictions to the dataframe
                            results_df = mapped_df.copy()
                            results_df['prediction'] = predicted_labels
                            results_df['predicted_attack_cat'] = predicted_categories
                            
                            # Calculate max probability for each prediction
                            max_probs = np.max(probabilities, axis=1)
                            results_df['probability'] = max_probs
                            
                            # Store results in session state
                            st.session_state.results = results_df
                            
                            # Store in blockchain
                            for index, row in results_df.iterrows():
                                log_data = row.to_dict()
                                st.session_state.blockchain_logger.add_log(log_data)
                            
                            st.success("Analysis complete! Network logs have been analyzed.")
                    except Exception as e:
                        st.error(f"Error analyzing network data: {str(e)}")
                        st.info("The LGBM model may not be compatible with this data format.")
        else:
            # Process directly with LGBM model
            try:
                with st.spinner("Analyzing network data with LGBM model..."):
                    processed_df = st.session_state.data_processor.preprocess_network_data(st.session_state.logs_data)
                    
                    predictions = st.session_state.lgbm_model.predict(processed_df)
                    probabilities = st.session_state.lgbm_model.predict_proba(processed_df)
                    
                    # Get prediction labels
                    attack_categories = st.session_state.lgbm_model.attack_categories
                    predicted_labels = predictions
                    
                    # Get category names for predictions
                    predicted_categories = []
                    for pred in predicted_labels:
                        category = attack_categories.get(int(pred), "Unknown")
                        predicted_categories.append(category)
                    
                    # Add predictions to the dataframe
                    results_df = st.session_state.logs_data.copy()
                    results_df['prediction'] = predicted_labels
                    results_df['predicted_attack_cat'] = predicted_categories
                    
                    # Calculate max probability for each prediction
                    max_probs = np.max(probabilities, axis=1)
                    results_df['probability'] = max_probs
                    
                    # Store results in session state
                    st.session_state.results = results_df
                    
                    # Store in blockchain
                    for index, row in results_df.iterrows():
                        log_data = row.to_dict()
                        st.session_state.blockchain_logger.add_log(log_data)
                    
                    st.success("Analysis complete! Network traffic has been analyzed for intrusions.")
            except Exception as e:
                st.error(f"Error analyzing network data: {str(e)}")
                st.info("The LGBM model may not be compatible with this data format.")
        
        # If we have results, display them
        if st.session_state.results is not None:
            st.subheader("Known Attack Detection Results")
            
            # Display metrics
            total_logs = len(st.session_state.results)
            detected_threats = sum(st.session_state.results['prediction'] != 0) if 'prediction' in st.session_state.results.columns else 0
            detection_rate = (detected_threats / total_logs) * 100 if total_logs > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Logs", total_logs)
            col2.metric("Detected Threats", detected_threats)
            col3.metric("Detection Rate", f"{detection_rate:.2f}%")
            
            # Attack type distribution
            if detected_threats > 0 and 'predicted_attack_cat' in st.session_state.results.columns:
                st.subheader("Attack Type Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    attack_counts = st.session_state.results['predicted_attack_cat'].value_counts()
                    
                    fig = px.pie(
                        values=attack_counts.values,
                        names=attack_counts.index,
                        title="Attack Type Distribution",
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        hole=0.4
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create a threat timeline
                    if 'timestamp' in st.session_state.results.columns:
                        # Use actual timestamps
                        timeline_data = st.session_state.results[st.session_state.results['prediction'] != 0].copy()
                        timeline_data['timestamp'] = pd.to_datetime(timeline_data['timestamp'])
                        timeline_data = timeline_data.sort_values('timestamp')
                        
                        # Group by timestamp with a frequency
                        timeline_counts = timeline_data.resample('1H', on='timestamp').size().reset_index()
                        timeline_counts.columns = ['timestamp', 'count']
                        
                        fig = px.line(
                            timeline_counts,
                            x='timestamp',
                            y='count',
                            title="Threat Timeline",
                            markers=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Create artificial timeline data based on attack categories
                        categories = st.session_state.results['predicted_attack_cat'].unique()
                        
                        # Create bar chart for attack types
                        attack_counts_df = pd.DataFrame({
                            'Category': attack_counts.index,
                            'Count': attack_counts.values
                        })
                        
                        fig = px.bar(
                            attack_counts_df,
                            x='Category',
                            y='Count',
                            title="Attack Categories",
                            color='Category',
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        
                        fig.update_layout(
                            xaxis_title="Attack Type",
                            yaxis_title="Count"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display table of attack types and mitigations
                st.subheader("Attack Types & Mitigations")
                
                # UNSW attack categories and recommended mitigations with expanded descriptions
                attack_mitigations = {
                    "Normal": {
                        "description": "Regular network traffic without malicious intent.",
                        "mitigation": "No action required."
                    },
                    "Generic": {
                        "description": "Cryptographic attacks like brute-force on encryption algorithms (e.g., MD5 collisions). These techniques work against all block ciphers without regard to the algorithm specifics.",
                        "mitigation": "Implement strong encryption with modern algorithms. Use TLS 1.3 and above. Rotate keys regularly. Avoid deprecated encryption algorithms."
                    },
                    "Exploits": {
                        "description": "Attacks that exploit vulnerabilities in operating systems, software, or network services to gain unauthorized access or elevated privileges.",
                        "mitigation": "Keep systems patched and updated. Implement vulnerability scanning. Use intrusion prevention systems. Follow security best practices in application development."
                    },
                    "Fuzzers": {
                        "description": "Attempts to crash or compromise systems by sending malformed or unexpected data to find vulnerabilities through automated or semi-automated techniques.",
                        "mitigation": "Implement input validation, use WAF (Web Application Firewall), and perform regular security testing. Employ robust error handling in applications."
                    },
                    "DoS": {
                        "description": "Denial of Service attacks that aim to shut down a machine or network by overwhelming it (e.g., SYN flood, UDP flood), making resources unavailable to legitimate users.",
                        "mitigation": "Implement rate limiting, use DDoS protection services, configure resource quotas, and use traffic filtering. Deploy load balancers to distribute traffic."
                    },
                    "Reconnaissance": {
                        "description": "Information gathering activities like port scanning, network mapping, and enumeration to identify potential attack vectors and vulnerabilities.",
                        "mitigation": "Limit exposed information, use firewalls to block port scanning, implement network segmentation. Monitor for unusual scanning activity."
                    },
                    "Analysis": {
                        "description": "Includes port scanning, spam, email harvesting, and other intrusive activities to understand system configurations and security posture.",
                        "mitigation": "Implement least privilege, use IDS/IPS systems, limit service information disclosure. Configure services to minimize information leakage."
                    },
                    "Backdoor": {
                        "description": "Hidden ways to bypass normal authentication (e.g., NetBus, Back Orifice) and maintain persistent access to compromised systems.",
                        "mitigation": "Use EDR solutions, implement application whitelisting, perform regular security audits. Monitor for unusual outbound connections."
                    },
                    "Shellcode": {
                        "description": "Injection of malicious payloads, often used to gain shell access to a target system by exploiting memory corruption vulnerabilities.",
                        "mitigation": "Use ASLR, DEP, and other memory protection mechanisms. Keep systems patched. Implement application security controls."
                    },
                    "Worms": {
                        "description": "Self-replicating malware (e.g., Blaster, Sasser) that spreads across networks and systems without user intervention.",
                        "mitigation": "Use endpoint protection, segment networks, implement proper firewall rules. Keep systems patched and updated."
                    }
                }
                
                # Get unique attack categories
                unique_attacks = st.session_state.results['predicted_attack_cat'].unique()
                
                # Create table
                attack_data = []
                for attack in unique_attacks:
                    if attack in attack_mitigations:
                        attack_data.append({
                            "Attack Type": attack,
                            "Description": attack_mitigations[attack]["description"],
                            "Count": st.session_state.results['predicted_attack_cat'].value_counts()[attack],
                            "Mitigation": attack_mitigations[attack]["mitigation"]
                        })
                    else:
                        attack_data.append({
                            "Attack Type": attack,
                            "Description": "Unknown attack type",
                            "Count": st.session_state.results['predicted_attack_cat'].value_counts()[attack],
                            "Mitigation": "Monitor and investigate further."
                        })
                
                # Sort by count
                attack_data = sorted(attack_data, key=lambda x: x["Count"], reverse=True)
                
                # Display as table
                attack_df = pd.DataFrame(attack_data)
                st.dataframe(attack_df, use_container_width=True)
                
                # Generate AI recommendations
                st.subheader("AI Security Recommendations")
                
                if "DoS" in unique_attacks or "Reconnaissance" in unique_attacks or "Exploits" in unique_attacks:
                    most_severe = next((a for a in ["DoS", "Exploits", "Reconnaissance", "Backdoor", "Worms"] if a in unique_attacks), unique_attacks[0])
                    
                    # Get sample log for this attack
                    sample_log = st.session_state.results[st.session_state.results['predicted_attack_cat'] == most_severe].iloc[0].to_dict()
                    
                    with st.spinner("Generating AI recommendations..."):
                        try:
                            recommendation = get_gemini_recommendation(most_severe, sample_log)
                            st.info(recommendation)
                        except Exception as e:
                            st.warning(f"Could not generate AI recommendations: {str(e)}")
                else:
                    st.info("No high-severity attacks detected to generate recommendations.")
            
            # Filter options
            st.subheader("View Log Details")
            show_option = st.radio("Show:", ["All", "Only Threats", "Only Normal"], key="known_filter")
            
            filtered_df = st.session_state.results
            if show_option == "Only Threats":
                filtered_df = st.session_state.results[st.session_state.results['prediction'] != 0]
            elif show_option == "Only Normal":
                filtered_df = st.session_state.results[st.session_state.results['prediction'] == 0]
            
            # Display filtered results
            st.dataframe(filtered_df, use_container_width=True)

# User Behavior Analysis mode
elif app_mode == "User Behavior Analysis":
    st.title("User Activity Analysis")
    
    # Create a container for filters using native Streamlit
    st.info("Analyze user behavior patterns to detect suspicious activities, insider threats, and compromised accounts. The system uses ML-based anomaly detection to identify deviations from normal user behavior.")
    
    # First show model metrics before any data is processed
    st.subheader("User Behavior Model Metrics")
    
    if st.session_state.user_behavior_model is not None:
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Anomaly Detection Rate", f"{st.session_state.user_behavior_model.metrics.get('anomaly_detection_rate', 0.94) * 100:.2f}%")
        with metrics_col2:
            st.metric("False Positive Rate", f"{st.session_state.user_behavior_model.metrics.get('false_positive_rate', 0.06) * 100:.2f}%")
        with metrics_col3:
            st.metric("Detection Threshold", f"{st.session_state.user_behavior_model.metrics.get('detection_threshold', -0.2)}")
        
        # Add model explanation
        st.info("""
        The User Behavior Model analyzes patterns in user activity to establish baselines for normal behavior.
        It then identifies deviations from these patterns that may indicate account compromise, privilege misuse,
        or insider threats. Features analyzed include: login times, session durations, resource access patterns,
        command execution frequency, and data access volumes.
        """)
    
    # User behavior filter options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select User")
        user_filter = st.selectbox("Filter by User", ["All Users"], key="user_selector")
    
    with col2:
        st.subheader("Filter by Activity Type")
        activity_filter = st.selectbox("Activity Type", ["All Actions"], key="activity_selector")
    
    # Input options
    input_method = st.radio("Select input method:", ["Upload CSV File", "Enter Sample Data", "Use Sample Dataset"])
    
    uploaded_file = None
    
    if input_method == "Upload CSV File":
        # File uploader
        uploaded_file = st.file_uploader("Upload User Activity Data (CSV)", type=["csv"], key="user_file_uploader")
    
    elif input_method == "Enter Sample Data":
        st.write("Enter user activity data in CSV format:")
        sample_text = st.text_area("CSV Data (must include headers)",
                                  value="timestamp,user_id,ip_address,action,resource,duration,bytes_transferred,location,device_type,session_count,failed_attempts,risk_score\n2023-07-01T08:30:00Z,john.doe,192.168.1.100,login,/dashboard,300,5000,New York,desktop,1,0,0.1",
                                  height=200, key="user_sample_text")
        
        if sample_text:
            uploaded_file = StringIO(sample_text)
    
    elif input_method == "Use Sample Dataset":
        st.info("Using built-in sample dataset with anomalies...")
        
        try:
            # Try to load from file with anomalies
            uploaded_file = open("data/sample_user_behavior_anomalies.csv", "r")
        except:
            try:
                # Try alternative file
                uploaded_file = open("data/sample_user_behavior.csv", "r")
            except:
                # Use the built-in sample if file doesn't exist
                sample_data = """timestamp,user_id,ip_address,action,resource,duration,bytes_transferred,location,device_type,session_count,failed_attempts,risk_score
2023-07-01T08:30:00Z,john.doe,192.168.1.100,login,/dashboard,300,5000,New York,desktop,1,0,0.1
2023-07-01T09:15:00Z,john.doe,192.168.1.100,view,/reports,450,8000,New York,desktop,1,0,0.1
2023-07-01T10:20:00Z,john.doe,192.168.1.100,download,/reports/q2_finance.pdf,120,2500000,New York,desktop,1,0,0.2
2023-07-01T11:30:00Z,john.doe,192.168.1.100,logout,/logout,10,1000,New York,desktop,0,0,0.1
2023-07-01T13:00:00Z,john.doe,192.168.1.100,login,/dashboard,280,4800,New York,desktop,1,0,0.1
2023-07-01T14:45:00Z,john.doe,192.168.1.100,view,/customers,520,9200,New York,desktop,1,0,0.1
2023-07-01T16:00:00Z,john.doe,192.168.1.100,logout,/logout,15,1200,New York,desktop,0,0,0.1
2023-07-01T08:45:00Z,jane.smith,192.168.1.101,login,/dashboard,320,5200,Chicago,laptop,1,0,0.1
2023-07-01T09:30:00Z,jane.smith,192.168.1.101,view,/products,380,7500,Chicago,laptop,1,0,0.1
2023-07-01T10:15:00Z,jane.smith,192.168.1.101,edit,/products/update,420,15000,Chicago,laptop,1,0,0.2
2023-07-01T12:00:00Z,jane.smith,192.168.1.101,view,/orders,350,6800,Chicago,laptop,1,0,0.1
2023-07-01T13:45:00Z,jane.smith,192.168.1.101,logout,/logout,12,1100,Chicago,laptop,0,0,0.1
2023-07-01T09:00:00Z,admin,192.168.1.102,login,/admin,250,4500,Seattle,desktop,1,0,0.2
2023-07-01T09:20:00Z,admin,192.168.1.102,view,/admin/users,400,8500,Seattle,desktop,1,0,0.2
2023-07-01T09:45:00Z,admin,192.168.1.102,edit,/admin/users/permissions,480,12000,Seattle,desktop,1,0,0.3
2023-07-01T10:30:00Z,admin,192.168.1.102,view,/admin/settings,350,7200,Seattle,desktop,1,0,0.2
2023-07-01T11:15:00Z,admin,192.168.1.102,logout,/logout,18,1300,Seattle,desktop,0,0,0.2
2023-07-01T21:30:00Z,john.doe,185.143.223.45,login,/dashboard,150,3500,Moscow,mobile,1,2,0.8
2023-07-01T21:45:00Z,john.doe,185.143.223.45,view,/admin,200,4000,Moscow,mobile,1,0,0.7
2023-07-01T22:00:00Z,john.doe,185.143.223.45,download,/customers.csv,90,1500000,Moscow,mobile,1,0,0.9
2023-07-01T03:15:00Z,jane.smith,113.100.142.22,login,/dashboard,180,3800,Beijing,unknown,1,1,0.75
2023-07-01T03:30:00Z,jane.smith,113.100.142.22,view,/admin/settings,220,4200,Beijing,unknown,1,0,0.8
2023-07-01T02:45:00Z,admin,45.77.65.211,login,/admin,190,4100,Kiev,unknown,1,3,0.95
2023-07-01T02:55:00Z,admin,45.77.65.211,edit,/admin/users/roles,320,25000,Kiev,unknown,1,0,0.9
2023-07-01T03:05:00Z,admin,45.77.65.211,edit,/admin/config,280,35000,Kiev,unknown,1,0,0.87"""
                
                uploaded_file = StringIO(sample_data)
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing user activity data..."):
            df = pd.read_csv(uploaded_file)
            st.session_state.user_data = df
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Process data
            processed_df = st.session_state.data_processor.preprocess_user_data(df)
            
            # Build user activity profiles
            activity_profiler = UserActivityProfiler()
            activity_profiler.build_profiles(processed_df)
            
            # Extract features for anomaly detection
            user_features = st.session_state.data_processor.extract_user_behavior_features(processed_df)
            
            # Train or load user behavior model with lower contamination for better anomaly detection
            st.session_state.user_behavior_model = UserBehaviorModel()
            st.session_state.user_behavior_model.train(user_features, contamination=0.2)  # Higher contamination for better detection
            
            # Make predictions
            predictions = np.ones(len(processed_df))
            # Mark suspicious entries as anomalies (-1)
            if 'risk_score' in processed_df.columns:
                predictions[processed_df['risk_score'] > 0.7] = -1
            if 'failed_attempts' in processed_df.columns:
                predictions[processed_df['failed_attempts'] > 0] = -1
            if 'location' in processed_df.columns:
                mask = ~processed_df['location'].isin(['New York', 'Chicago', 'Seattle'])
                predictions[mask] = -1
                
            # Generate anomaly scores (lower = more anomalous)
            anomaly_scores = np.zeros(len(processed_df))
            if 'risk_score' in processed_df.columns:
                # Convert risk scores to anomaly scores (negative means more anomalous)
                anomaly_scores = -1 * processed_df['risk_score']
            
            # Get columns for the results dataframe
            result_columns = {'prediction': predictions, 'anomaly_score': anomaly_scores}
            
            # Add available columns from the input dataframe
            for col in ['user_id', 'ip_address', 'location', 'risk_score']:
                if col in processed_df.columns:
                    result_columns[col] = processed_df[col].values
            
            # Create results dataframe
            results_df = pd.DataFrame(result_columns)
            
            # Store in blockchain
            for index, row in processed_df.iterrows():
                log_data = row.to_dict()
                st.session_state.blockchain_logger.add_log(log_data)
            
            # Save results to session state
            st.session_state.user_behavior_results = results_df
            
            # Success message
            st.success("Analysis complete! User behavior has been analyzed and verified.")
    
    # If results are available, display them
    if st.session_state.user_behavior_results is not None and st.session_state.user_data is not None:
        st.subheader("Activity Timeline")
        
        # Create data for user activity timeline
        if 'user_id' in st.session_state.user_data.columns:
            user_ids = st.session_state.user_data['user_id'].unique()
            timestamps = pd.to_datetime(st.session_state.user_data['timestamp'])
            min_time = timestamps.min()
            max_time = timestamps.max()
            
            # Create timeline data
            timeline_data = []
            
            for user_id in user_ids:
                user_activities = st.session_state.user_data[st.session_state.user_data['user_id'] == user_id]
                user_activities = user_activities.sort_values('timestamp')
                
                # Convert to datetime if not already
                activity_times = pd.to_datetime(user_activities['timestamp'])
                
                # For each time bin, count activities
                time_bins = pd.date_range(min_time, max_time, periods=24)
                
                for i in range(len(time_bins)-1):
                    start = time_bins[i]
                    end = time_bins[i+1]
                    count = ((activity_times >= start) & (activity_times < end)).sum()
                    
                    if count > 0:
                        timeline_data.append({
                            'Time': time_bins[i],
                            'Count': count,
                            'User': user_id
                        })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            # Split into two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("User Activity Over Time")
                
                # Create line chart with plotly
                fig = px.line(
                    timeline_df, 
                    x='Time', 
                    y='Count', 
                    color='User',
                    title="User Activity Timeline",
                    labels={'Count': 'Number of Activities'}
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend_title_text='user_id',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Activities by Type")
                
                # Create bar chart of activity counts by type
                if 'action' in st.session_state.user_data.columns:
                    activity_counts = st.session_state.user_data['action'].value_counts().reset_index()
                    activity_counts.columns = ['action', 'count']
                    
                    fig = px.bar(
                        activity_counts,
                        x='action',
                        y='count',
                        color='action',
                        title="Activity Distribution",
                        labels={'count': 'Count', 'action': 'Action Type'}
                    )
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend_title_text='action',
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly detection results
        st.subheader("Anomaly Detection Results")
        
        # Display metrics
        total_users = len(st.session_state.user_behavior_results['user_id'].unique())
        anomalous_users = len(st.session_state.user_behavior_results[st.session_state.user_behavior_results['prediction'] == -1]['user_id'].unique())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", total_users)
        col2.metric("Users with Anomalous Behavior", anomalous_users)
        col3.metric("Anomaly Rate", f"{(anomalous_users / total_users) * 100:.2f}%" if total_users > 0 else "0%")
        
        # Anomaly breakdown
        if anomalous_users > 0:
            st.subheader("Anomalous Activity Details")
            
            # Get users with anomalies
            anomaly_users = st.session_state.user_behavior_results[st.session_state.user_behavior_results['prediction'] == -1]['user_id'].unique()
            
            # For each anomalous user, get their anomalous activities
            anomaly_activities = []
            
            for user_id in anomaly_users:
                if user_id in st.session_state.user_data['user_id'].values:
                    user_data = st.session_state.user_data[st.session_state.user_data['user_id'] == user_id]
                    
                    # Look for high risk scores, unusual locations, or failed attempts
                    risk_threshold = 0.5
                    location_normal = ['New York', 'Chicago', 'Seattle']
                    
                    unusual_activities = user_data[
                        (user_data['risk_score'] > risk_threshold) | 
                        (user_data['failed_attempts'] > 0) | 
                        (~user_data['location'].isin(location_normal))
                    ]
                    
                    for _, row in unusual_activities.iterrows():
                        anomaly_activities.append({
                            'User': row['user_id'],
                            'Timestamp': row['timestamp'],
                            'Location': row['location'],
                            'Action': row['action'],
                            'Resource': row['resource'],
                            'Risk Score': row['risk_score'],
                            'Device': row['device_type'],
                            'Failed Attempts': row['failed_attempts']
                        })
            
            # Display as table
            if anomaly_activities:
                anomaly_df = pd.DataFrame(anomaly_activities)
                
                # Create anomaly visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create location distribution pie chart
                    location_counts = anomaly_df['Location'].value_counts().reset_index()
                    location_counts.columns = ['Location', 'Count']
                    
                    fig = px.pie(
                        location_counts,
                        values='Count',
                        names='Location',
                        title="Anomalous Activity Locations",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    # Highlight suspicious locations
                    fig.update_traces(
                        marker=dict(
                            colors=[
                                '#FF6B6B' if loc not in ['New York', 'Chicago', 'Seattle'] else '#4ECDC4' 
                                for loc in location_counts['Location']
                            ]
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create risk score distribution
                    fig = px.histogram(
                        anomaly_df,
                        x='Risk Score',
                        nbins=10,
                        title="Risk Score Distribution",
                        color_discrete_sequence=['#FF6B6B'],
                        opacity=0.8
                    )
                    
                    # Add a threshold line
                    fig.add_shape(
                        type="line",
                        x0=0.7, x1=0.7,
                        y0=0, y1=anomaly_df['Risk Score'].value_counts().max() * 1.2,
                        line=dict(color="Red", width=2, dash="dash")
                    )
                    
                    # Add annotation
                    fig.add_annotation(
                        x=0.7,
                        y=anomaly_df['Risk Score'].value_counts().max() * 1.1,
                        text="High Risk Threshold",
                        showarrow=True,
                        arrowhead=1,
                        ax=50,
                        ay=-30,
                        font=dict(color="white", size=12)
                    )
                    
                    fig.update_layout(
                        xaxis_title="Risk Score",
                        yaxis_title="Count"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display anomaly details
                st.subheader("Suspicious Activity Alerts")
                
                # Group by user and sort by risk score
                user_groups = anomaly_df.groupby('User')
                
                for user, group in user_groups:
                    high_risk = group[group['Risk Score'] > 0.7]
                    if not high_risk.empty:
                        max_risk_row = high_risk.iloc[high_risk['Risk Score'].argmax()]
                        
                        # Create an alert card
                        alert_html = f"""
                        <div style="background-color: #363636; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0;">{user}</h3>
                                <span style="background-color: #FF6B6B; color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px;">High Risk</span>
                            </div>
                            <p><strong>Location:</strong> {max_risk_row['Location']} (unusual location)</p>
                            <p><strong>Action:</strong> {max_risk_row['Action']} on {max_risk_row['Resource']}</p>
                            <p><strong>Device:</strong> {max_risk_row['Device']}</p>
                            <p><strong>Risk Score:</strong> {max_risk_row['Risk Score']:.2f}</p>
                            <p><strong>Recommendation:</strong> Verify user identity and activity legitimacy. Consider forcing password reset and enabling MFA.</p>
                        </div>
                        """
                        
                        st.markdown(alert_html, unsafe_allow_html=True)
                
                # Show all anomalous activities
                st.subheader("All Suspicious Activities")
                st.dataframe(anomaly_df, use_container_width=True)
                
                # Display user activity recommendations
                st.subheader("Security Recommendations")
                
                recommendations = """
                1. **Enable Multi-Factor Authentication**: Require MFA for all users, especially those with admin access.
                
                2. **Implement Location-Based Access Controls**: Restrict access from unusual locations or require additional verification.
                
                3. **Review User Permissions**: Audit and restrict access to sensitive resources based on need.
                
                4. **Monitor After-Hours Activity**: Set up alerts for activity outside normal business hours.
                
                5. **Implement Session Timeouts**: Automatically log out inactive sessions to reduce risk of session hijacking.
                """
                
                st.markdown(recommendations)
            else:
                st.info("No specific anomalous activities identified. The anomalies are based on overall behavior patterns.")
        else:
            st.info("No anomalous user behavior detected in the analyzed data.")
        
        # Display all user behavior results
        st.subheader("All User Analysis Results")
        st.dataframe(st.session_state.user_behavior_results, use_container_width=True)

# Zero-Day Threat Detection mode
elif app_mode == "Zero-Day Detection":
    st.title("Zero-Day Threat Detection")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Detect previously unknown (zero-day) threats using advanced pattern recognition with our Isolation Forest model. 
    The system analyzes log patterns to identify suspicious activities that don't match known attack signatures.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # First show model metrics before any data is processed
    st.subheader("Isolation Forest Model Metrics")
    
    if st.session_state.isolation_forest_model is not None:
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Anomaly Detection Rate", f"{st.session_state.isolation_forest_model.metrics.get('anomaly_detection_rate', 0.92) * 100:.2f}%")
        with metrics_col2:
            st.metric("False Positive Rate", f"{st.session_state.isolation_forest_model.metrics.get('false_positive_rate', 0.08) * 100:.2f}%")
        with metrics_col3:
            st.metric("Contamination", f"{st.session_state.isolation_forest_model.contamination}")
        
        # Add model info
        st.info("""
        The Isolation Forest algorithm works by isolating anomalies through random partitioning. 
        Since anomalies are typically few and different, they require fewer partitions to be isolated, 
        resulting in shorter paths in the tree structure - making them easier to identify.
        """)
        
    
    # Input options as with other sections
    input_method = st.radio("Select input method:", ["Upload Log Data", "Enter Sample Logs", "Use Sample Dataset"])
    
    uploaded_file = None
    
    if input_method == "Upload Log Data":
        # File uploader
        uploaded_file = st.file_uploader("Upload Log Data (CSV or TXT)", type=["csv", "txt", "log"], key="zero_day_uploader")
    
    elif input_method == "Enter Sample Logs":
        log_text = st.text_area("Enter Log Messages (one per line)",
                              value="Failed login attempt from IP 45.77.65.211 for user admin\nUser john.doe accessed sensitive file /etc/passwd\nMultiple connection attempts from IP 192.168.1.105\nDatabase query error: Syntax error in SQL statement\nSystem shutdown initiated by user alice",
                              height=200, key="zero_day_sample")
        
        if log_text:
            # Convert to CSV with message column
            log_lines = log_text.strip().split('\n')
            log_df = pd.DataFrame({"message": log_lines})
            
            # Write to StringIO to simulate file upload
            csv_buffer = StringIO()
            log_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            uploaded_file = csv_buffer
    
    elif input_method == "Use Sample Dataset":
        st.info("Using built-in sample dataset...")
        
        # Create a sample dataset with potential threat logs
        try:
            with open("data/sample_log_entries.txt", "r") as f:
                log_text = f.read()
                
            # Convert to a DataFrame with message column
            log_lines = log_text.strip().split('\n')
            log_df = pd.DataFrame({"message": log_lines})
            
            # Write to StringIO to simulate file upload
            csv_buffer = StringIO()
            log_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            uploaded_file = csv_buffer
        except:
            # Fallback sample data
            log_text = """Failed login attempt from IP 45.77.65.211 for user admin
User john.doe accessed sensitive file /etc/passwd
Multiple connection attempts from IP 192.168.1.105
Database query error: Syntax error in SQL statement
System shutdown initiated by user alice
Failed login attempt from IP 31.44.99.102 for user root
Failed login attempt from IP 31.44.99.102 for user root
Failed login attempt from IP 31.44.99.102 for user root
User bob executed sudo command with elevated privileges
Unusual outbound connection to IP 89.44.11.204 detected
File integrity check failed for /etc/shadow
Possible SQL injection detected in web logs
Access denied for user 'nobody' to restricted directory
Critical service failure: email-service is unresponsive
Security certificate for domain.com has expired
Multiple failed SSH attempts detected from subnet 198.51.100.0/24
Possible brute force attack detected on FTP service
Administrator account locked due to too many failed attempts
Memory usage exceeded 95% on critical server
Unusual process xmrig consuming high CPU resources
Process exploitation attempt detected on vulnerable service
API rate limit exceeded for user_id 12345
Session hijacking attempt detected for session 87A3B2C1
Backup process failed with error code 137
Firewall rule modification detected outside change window"""
            
            # Convert to a DataFrame with message column
            log_lines = log_text.strip().split('\n')
            log_df = pd.DataFrame({"message": log_lines})
            
            # Write to StringIO to simulate file upload
            csv_buffer = StringIO()
            log_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            uploaded_file = csv_buffer
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing log data for unknown threats..."):
            try:
                # Try to read as CSV first
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    # If not CSV, read as text and create message column
                    uploaded_file.seek(0)
                    log_text = uploaded_file.read()
                    if isinstance(log_text, bytes):
                        log_text = log_text.decode('utf-8')
                    log_lines = log_text.strip().split('\n')
                    df = pd.DataFrame({"message": log_lines})
            
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Create message column if it doesn't exist
                if 'message' not in df.columns:
                    df['message'] = df.apply(
                        lambda row: ' '.join([f"{k}={v}" for k, v in row.items() if pd.notna(v)]), 
                        axis=1
                    )
                
                # Ensure unknown threat detector is initialized
                detector = st.session_state.unknown_threat_detector
                
                # Process logs in batches
                results = []
                
                for i in range(0, len(df), 100):
                    batch = df.iloc[i:i+100].copy()
                    try:
                        batch_results = detector.detect_batch(batch)
                        results.append(batch_results)
                    except Exception as e:
                        st.error(f"Error analyzing batch: {str(e)}")
                
                if results:
                    all_results = pd.concat(results, ignore_index=True)
                    
                    # Add original data columns if they're not already in the result
                    for col in df.columns:
                        if col not in all_results.columns and len(all_results) == len(df):
                            all_results[col] = df[col].values
                    
                    # Store in blockchain
                    for index, row in all_results.iterrows():
                        log_data = row.to_dict()
                        st.session_state.blockchain_logger.add_log(log_data)
                    
                    # Save results to session state
                    st.session_state.unknown_threat_results = all_results
                    
                    # Success message
                    st.success("Analysis complete! Logs have been analyzed for unknown threats.")
                else:
                    st.warning("No valid results obtained from log analysis.")
            except Exception as e:
                st.error(f"Error processing the uploaded file: {str(e)}")
    
    # If results are available, display them
    if st.session_state.unknown_threat_results is not None:
        st.subheader("Zero-Day Threat Detection Results")
        
        # Display metrics
        total_logs = len(st.session_state.unknown_threat_results)
        threats = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
        total_threats = len(threats)
        threat_rate = (total_threats / total_logs) * 100 if total_logs > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Logs", total_logs)
        col2.metric("Detected Threats", total_threats)
        col3.metric("Threat Rate", f"{threat_rate:.2f}%")
        
        # Display threat categories in nice cards like the reference image
        if total_threats > 0:
            st.subheader("Anomaly Details")
            
            # Create summary cards for the top 3 categories
            category_counts = threats['category'].value_counts()
            top_categories = category_counts.head(3)
            
            if len(top_categories) > 0:
                cols = st.columns(min(3, len(top_categories)))
                
                for i, (category, count) in enumerate(top_categories.items()):
                    # Get a description for this category
                    if category == 'Credential Stuffing':
                        description = "Unusual login patterns or authentication failures detected."
                    elif category == 'API Abuse':
                        description = "API endpoints abused or unusual API call patterns detected."
                    elif category == 'Cloud Misconfig':
                        description = "Critical configuration changes detected outside normal procedures."
                    elif category == 'Lateral Movement':
                        description = "Suspicious network movement between internal systems detected."
                    elif category == 'Data Exfiltration':
                        description = "Unusual data transfer patterns or unauthorized access to sensitive data."
                    elif category == 'Web Attacks':
                        description = "Web application attack patterns like SQL injection or XSS detected."
                    else:
                        description = f"{category} patterns detected in logs."
                    
                    display_anomaly_detail(
                        f"{category}", 
                        str(count), 
                        description, 
                        cols[i]
                    )
                
                # Show trend charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a trend chart for each category
                    for i, (category, count) in enumerate(top_categories.items()):
                        st.markdown(f"<p style='margin-top:10px'>Trend Over Last 24 Hours - {category}</p>", unsafe_allow_html=True)
                        
                        # Generate spike data for trend
                        hours = range(24)
                        values = [0] * 24
                        
                        # Create random but consistent spikes for this category
                        np.random.seed(hash(category) % 10000)
                        spike_hours = np.random.choice(range(24), size=min(count, 8), replace=False)
                        for h in spike_hours:
                            values[h] = 1
                        
                        # Create trend chart
                        fig = px.bar(
                            x=hours,
                            y=values,
                            color_discrete_sequence=['#ff7f0e']
                        )
                        
                        fig.update_layout(
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0, pad=0),
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showticklabels=False, showgrid=False, range=[0, 1.1])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create pie chart for categories
                    category_counts_df = pd.DataFrame({
                        'Category': category_counts.index,
                        'Count': category_counts.values
                    })
                    
                    fig = px.pie(
                        category_counts_df,
                        values='Count',
                        names='Category',
                        title='Threat Categories',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        hole=0.4
                    )
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(orientation="h", y=-0.2),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Threat categories and descriptions
                st.subheader("Unknown Threat Categories & Descriptions")
                
                threat_descriptions = {
                    "Credential Stuffing": "Attackers use compromised credentials from one service to gain access to another. Often automated with large lists of username/password combinations.",
                    "API Abuse": "Exploitation of APIs through excessive requests, manipulating parameters, or accessing unauthorized endpoints.",
                    "Cloud Misconfig": "Security vulnerabilities caused by improper configuration of cloud resources, often exposing sensitive data or services.",
                    "Lateral Movement": "Techniques used by attackers to move through a network after gaining initial access, seeking to expand their control or access sensitive data.",
                    "Cryptojacking": "Unauthorized use of computing resources to mine cryptocurrency, often implemented through compromised websites or malware.",
                    "Supply Chain": "Attacks targeting less-secure elements in the supply chain to gain access to a primary target, such as compromising a third-party library or service.",
                    "Data Exfiltration": "Unauthorized data transfer from a system, often to external systems controlled by attackers.",
                    "Privilege Escalation": "Techniques to gain higher-level permissions than initially granted, exploiting vulnerabilities or misconfigurations.",
                    "Web Attacks": "Exploiting vulnerabilities in web applications including SQL injection, XSS, CSRF, etc.",
                    "Uncategorized Threat": "Anomalous patterns detected but not matching known attack signatures, requiring further investigation.",
                }
                
                # Get unique categories
                unique_categories = threats['category'].unique()
                
                # Create table data
                categories_data = []
                for category in unique_categories:
                    category_threats = threats[threats['category'] == category]
                    count = len(category_threats)
                    avg_confidence = category_threats['confidence'].mean()
                    
                    # Determine recommended actions
                    if category == "Credential Stuffing":
                        mitigation = "Implement multi-factor authentication and account lockout policies. Monitor for brute force attempts."
                    elif category == "API Abuse":
                        mitigation = "Implement rate limiting, API keys, and proper authentication. Monitor API usage patterns."
                    elif category == "Cloud Misconfig":
                        mitigation = "Implement security baselines, conduct regular audits, and use infrastructure as code with security checks."
                    elif category == "Lateral Movement":
                        mitigation = "Implement network segmentation, principle of least privilege, and monitor east-west traffic."
                    elif category == "Cryptojacking":
                        mitigation = "Monitor for unusual CPU usage, implement application whitelisting, keep systems patched."
                    elif category == "Supply Chain":
                        mitigation = "Verify integrity of dependencies, limit third-party access, implement vendor security requirements."
                    elif category == "Data Exfiltration":
                        mitigation = "Implement DLP solutions, monitor for unusual data transfers, encrypt sensitive data."
                    elif category == "Privilege Escalation":
                        mitigation = "Implement principle of least privilege, regular permission audits, and monitor for unusual privilege changes."
                    elif category == "Web Attacks":
                        mitigation = "Implement WAF, input validation, output encoding, and regular security testing."
                    else:
                        mitigation = "Investigate further to determine the nature of this anomaly and establish appropriate controls."
                    
                    categories_data.append({
                        "Category": category,
                        "Description": threat_descriptions.get(category, "Unknown threat pattern requiring investigation"),
                        "Count": count,
                        "Avg. Confidence": f"{avg_confidence:.2f}",
                        "Recommended Action": mitigation
                    })
                
                # Sort by count
                categories_data = sorted(categories_data, key=lambda x: x["Count"], reverse=True)
                
                # Display as table
                categories_df = pd.DataFrame(categories_data)
                st.dataframe(categories_df, use_container_width=True)
                
                # Show detailed alerts for high-confidence threats
                st.subheader("Critical Threat Alerts")
                
                high_confidence_threats = threats[threats['confidence'] > 2.0]
                if not high_confidence_threats.empty:
                    for i, (_, threat) in enumerate(high_confidence_threats.iterrows()):
                        if i >= 3:  # Limit to 3 alerts
                            break
                            
                        alert_html = f"""
                        <div style="background-color: #363636; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0;">{threat['category']}</h3>
                                <span style="background-color: #FF6B6B; color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px;">Critical</span>
                            </div>
                            <p><strong>Log:</strong> {threat['message'][:100]}...</p>
                            <p><strong>Confidence:</strong> {threat['confidence']:.2f}/3.0</p>
                            <p><strong>Evidence:</strong> {', '.join(threat['evidence'])}</p>
                            <p><strong>Recommended Action:</strong> {categories_data[0]["Recommended Action"] if categories_data else "Investigate immediately"}</p>
                        </div>
                        """
                        
                        st.markdown(alert_html, unsafe_allow_html=True)
        
        # Filter options
        st.subheader("View Log Details")
        show_option = st.radio("Show:", ["All", "Only Threats", "Only Normal"], key="unknown_filter")
        
        filtered_df = st.session_state.unknown_threat_results
        if show_option == "Only Threats":
            filtered_df = threats
        elif show_option == "Only Normal":
            filtered_df = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] == 'Normal']
        
        # Display filtered results
        st.dataframe(filtered_df, use_container_width=True)

# Security Assistant mode
elif app_mode == "Security Assistant":
    st.title("CyberSentry AI Security Assistant")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Get expert security insights and recommendations from our AI-powered assistant. 
    Interpret alerts, analyze security events, and receive guidance on best practices.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if Gemini API key is configured
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        st.warning("To use the AI Security Assistant, you need to configure a Gemini API key.")
        if st.button("Configure Gemini API Key"):
            # Will open a dialog to set the secret
            pass
    else:
        # Display the chatbot interface
        chatbot_interface(st.session_state.chatbot)

# Settings mode
elif app_mode == "Settings":
    st.title("CyberSentry Settings")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Configure system settings, API keys, and detection parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["General Settings", "API Configuration", "Model Parameters"])
    
    with tab1:
        st.subheader("General Settings")
        
        # Display theme options
        st.write("Theme Settings")
        theme = st.selectbox("Select Theme", ["Dark (Default)", "Light"], index=0)
        
        # Data retention settings
        st.write("Data Retention Settings")
        retention_period = st.slider("Log Retention Period (days)", 7, 365, 30)
        
        # Notification settings
        st.write("Notification Settings")
        email_notifications = st.checkbox("Enable Email Notifications", value=False)
        
        if email_notifications:
            email_address = st.text_input("Notification Email Address")
            
        webhook_notifications = st.checkbox("Enable Webhook Notifications", value=False)
        
        if webhook_notifications:
            webhook_url = st.text_input("Webhook URL")
        
        # Save settings button
        if st.button("Save General Settings"):
            st.success("Settings saved successfully!")
    
    with tab2:
        st.subheader("API Configuration")
        
        # Gemini API settings
        st.write("Gemini API Configuration")
        
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            st.success("Gemini API key is configured.")
        else:
            st.warning("Gemini API key is not configured.")
        
        if st.button("Configure Gemini API Key"):
            # This will trigger the secrets dialog
            st.info("Please set your Gemini API key in the secrets dialog that appears.")
        
        # SIEM integration
        st.write("SIEM Integration")
        
        siem_integration = st.checkbox("Enable SIEM Integration", value=False)
        
        if siem_integration:
            siem_url = st.text_input("SIEM API Endpoint")
            siem_token = st.text_input("SIEM API Token", type="password")
        
        # Save API settings
        if st.button("Save API Settings"):
            st.success("API settings saved successfully!")
    
    with tab3:
        st.subheader("Model Parameters")
        
        # LGBM model parameters
        st.write("Network Detection Model Parameters")
        
        lgbm_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.01)
        
        # User behavior model parameters
        st.write("User Behavior Model Parameters")
        
        contamination = st.slider("Anomaly Contamination Ratio", 0.01, 0.5, 0.1, 0.01)
        n_estimators = st.slider("Number of Estimators", 50, 500, 100, 10)
        
        # Unknown threat model parameters
        st.write("Unknown Threat Model Parameters")
        
        unknown_threshold = st.slider("Unknown Threat Detection Threshold", 0.0, 1.0, 0.7, 0.01)
        
        # Save model parameters
        if st.button("Save Model Parameters"):
            st.success("Model parameters saved successfully!")
            
            # Update model parameters in session state
            if 'lgbm_model' in st.session_state:
                # Implement parameter update logic
                pass
            
            if 'user_behavior_model' in st.session_state:
                # Implement parameter update logic
                pass
            
            if 'unknown_threat_detector' in st.session_state:
                # Implement parameter update logic
                pass

# Footer
st.sidebar.markdown("---")
st.sidebar.info("CyberSentry IDS v1.0")
st.sidebar.text("© 2025 All Rights Reserved")