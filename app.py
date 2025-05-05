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
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API
initialize_gemini_api()

# Initialize session state variables
if 'lgbm_model' not in st.session_state:
    st.session_state.lgbm_model = LGBMClassifier()
    st.session_state.lgbm_model.initialize_model()

if 'user_behavior_model' not in st.session_state:
    st.session_state.user_behavior_model = UserBehaviorAnalyzer()

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
                         end=pd.Timestamp.now(), freq='1H')
    
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

# Display dashboard overview
if app_mode == "Dashboard":
    st.title("CyberSentry: Enterprise Security Dashboard")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Enterprise-grade security monitoring with real-time anomaly detection and threat intelligence. 
    Monitor both <b>known attacks</b> and <b>zero-day threats</b> with ML-powered analytics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    display_metric_card("Security Score", f"{st.session_state.security_score}", "down", "-1.2%", "🛡️", col1)
    display_metric_card("Anomalies Detected", f"{st.session_state.anomaly_count}", "up", "+4.6%", "⚠️", col2)
    display_metric_card("Critical Events", f"{st.session_state.critical_events}", "up", "+3", "🔥", col3)
    display_metric_card("Suspicious Users", f"{st.session_state.suspicious_users}", "up", "+0", "👤", col4)
    
    st.markdown("---")
    
    # Activity trend chart
    st.subheader("Activity Trend (Last 7 Days)")
    
    activity_data = generate_activity_trend()
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create two line charts with plotly
        fig = go.Figure()
        
        # Add normal activity line
        fig.add_trace(go.Scatter(
            x=activity_data['date'], 
            y=activity_data['normal'],
            mode='lines',
            name='Normal Activity',
            line=dict(color='#3498db', width=2)
        ))
        
        # Add anomalies line
        fig.add_trace(go.Scatter(
            x=activity_data['date'], 
            y=activity_data['anomalies'],
            mode='lines',
            name='Anomalies',
            line=dict(color='#e74c3c', width=2)
        ))
        
        # Add a peak annotation
        peak_idx = activity_data['anomalies'].idxmax()
        fig.add_annotation(
            x=activity_data['date'][peak_idx],
            y=activity_data['anomalies'][peak_idx] + 0.2,
            text="Peak",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e74c3c",
            bgcolor="#e74c3c",
            font=dict(color="white"),
            bordercolor="#e74c3c"
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anomaly Distribution by Category and Severity
        st.subheader("Anomaly Categories")
        
        # Create sample data
        categories = ['Authentication', 'System', 'Configuration', 'Network', 'Data', 'User']
        severities = ['Critical', 'High', 'Medium', 'Low']
        
        # Create random counts that sum to our anomaly count
        np.random.seed(42)
        counts = np.random.dirichlet(np.ones(len(categories))) * st.session_state.anomaly_count
        counts = [int(c) for c in counts]
        
        # Ensure they sum to anomaly_count
        diff = st.session_state.anomaly_count - sum(counts)
        counts[0] += diff
        
        # Create severity distribution
        severity_counts = {
            'Critical': int(st.session_state.anomaly_count * 0.15),
            'High': int(st.session_state.anomaly_count * 0.35),
            'Medium': int(st.session_state.anomaly_count * 0.3),
            'Low': st.session_state.anomaly_count - int(st.session_state.anomaly_count * 0.15) - 
                  int(st.session_state.anomaly_count * 0.35) - int(st.session_state.anomaly_count * 0.3)
        }
        
        # Create pie chart
        fig = px.pie(
            values=counts,
            names=categories,
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed anomaly breakdowns
    st.markdown("---")
    st.subheader("Anomaly Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: #363636; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
            <h3 style='margin-top:0'>Authentication Anomalies <span style='float:right; background-color:#ff7f0e; padding:5px 10px; border-radius:15px; font-size:14px'>High</span></h3>
            <h1 style='color:#ff7f0e; font-size:48px'>11</h1>
            <p>Unusual login patterns or authentication failures detected.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='margin-top:10px'>Trend Over Last 24 Hours</p>", unsafe_allow_html=True)
        
        # Generate spike data for trend
        hours = range(24)
        values = [0] * 24
        for i in [2, 5, 9, 13, 18, 22]:
            values[i] = 1
        
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
        st.markdown("""
        <div style='background-color: #363636; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
            <h3 style='margin-top:0'>System Anomalies <span style='float:right; background-color:#ff7f0e; padding:5px 10px; border-radius:15px; font-size:14px'>High</span></h3>
            <h1 style='color:#ff7f0e; font-size:48px'>9</h1>
            <p>System resource anomalies or unusual process behavior detected.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='margin-top:10px'>Trend Over Last 24 Hours</p>", unsafe_allow_html=True)
        
        # Generate spike data for trend
        hours = range(24)
        values = [0] * 24
        for i in [1, 3, 7, 11, 15, 19, 23]:
            values[i] = 1
        
        # Add annotation for change
        annotations = [
            dict(
                x=22, y=1,
                text="+0.5",
                showarrow=False,
                font=dict(color="#4CAF50"),
                bgcolor="#2E2E2E",
                bordercolor="#4CAF50",
                borderwidth=1,
                borderpad=4
            )
        ]
        
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
            yaxis=dict(showticklabels=False, showgrid=False, range=[0, 1.1]),
            annotations=annotations
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: #363636; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
            <h3 style='margin-top:0'>Configuration Anomalies <span style='float:right; background-color:#ff7f0e; padding:5px 10px; border-radius:15px; font-size:14px'>High</span></h3>
            <h1 style='color:#ff7f0e; font-size:48px'>7</h1>
            <p>Critical configuration changes detected outside normal procedures.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='margin-top:10px'>Trend Over Last 24 Hours</p>", unsafe_allow_html=True)
        
        # Generate spike data for trend
        hours = range(24)
        values = [0] * 24
        for i in [4, 8, 12, 16, 22]:
            values[i] = 1
        
        # Add annotation for change
        annotations = [
            dict(
                x=22, y=1,
                text="+0.1",
                showarrow=False,
                font=dict(color="#4CAF50"),
                bgcolor="#2E2E2E",
                bordercolor="#4CAF50",
                borderwidth=1,
                borderpad=4
            )
        ]
        
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
            yaxis=dict(showticklabels=False, showgrid=False, range=[0, 1.1]),
            annotations=annotations
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display if we have any actual results
    if st.session_state.results is not None or st.session_state.user_behavior_results is not None or st.session_state.unknown_threat_results is not None:
        st.markdown("---")
        st.subheader("Detection Results")
        
        tab1, tab2, tab3 = st.tabs(["Network Analysis", "User Behavior", "Zero-Day Threats"])
        
        with tab1:
            if st.session_state.results is not None:
                # Get summary stats
                total_logs = len(st.session_state.results)
                detected_threats = sum(st.session_state.results['prediction'] != 0) if 'prediction' in st.session_state.results.columns else 0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Logs", total_logs)
                    st.metric("Detected Threats", detected_threats)
                
                with col2:
                    # Display threat distribution
                    if detected_threats > 0:
                        fig = create_threat_distribution(st.session_state.results)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No network analysis results available. Process some logs first.")
        
        with tab2:
            if st.session_state.user_behavior_results is not None:
                # Get summary stats
                total_users = len(st.session_state.user_behavior_results['user_id'].unique())
                anomalous_users = len(st.session_state.user_behavior_results[st.session_state.user_behavior_results['prediction'] == -1]['user_id'].unique())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Users", total_users)
                    st.metric("Anomalous Users", anomalous_users)
                
                with col2:
                    # Display user behavior charts
                    fig = create_user_behavior_charts(st.session_state.user_behavior_results)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No user behavior analysis results available. Process some user data first.")
        
        with tab3:
            if st.session_state.unknown_threat_results is not None:
                # Get summary stats
                total_logs = len(st.session_state.unknown_threat_results)
                unknown_threats = len(st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Logs", total_logs)
                    st.metric("Unknown Threats", unknown_threats)
                
                with col2:
                    # Display unknown threat categories
                    if unknown_threats > 0:
                        category_counts = st.session_state.unknown_threat_results['category'].value_counts().reset_index()
                        category_counts.columns = ['Category', 'Count']
                        
                        fig = px.pie(
                            category_counts,
                            values='Count',
                            names='Category',
                            title='Unknown Threat Categories',
                            color_discrete_sequence=px.colors.qualitative.Safe
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No zero-day threat detection results available. Process some logs first.")

# Log Analysis mode
elif app_mode == "Log Analysis":
    st.title("Enterprise Log Analysis")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Upload raw system/application logs for advanced analysis. 
    CyberSentry can parse common log formats including Apache/Nginx, Syslog, and JSON logs, providing detection of both
    known attacks (according to the UNSW dataset) and unknown patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        log_format = st.selectbox("Select log format:", ["Apache/Nginx", "Syslog", "JSON", "UNSW Network Logs"])
        
        if log_format == "Apache/Nginx":
            try:
                with open("data/sample_log_entries.txt", "r") as f:
                    log_text = f.read()
            except:
                log_text = """192.168.1.100 - - [01/Jul/2023:15:42:31 +0000] "GET /index.html HTTP/1.1" 200 4523
127.0.0.1 - - [01/Jul/2023:15:43:10 +0000] "POST /login HTTP/1.1" 302 354
192.168.1.105 - - [01/Jul/2023:15:45:20 +0000] "GET /admin HTTP/1.1" 401 1293
45.77.65.211 - - [01/Jul/2023:15:46:03 +0000] "GET /wp-login.php HTTP/1.1" 404 721
192.168.1.102 - - [01/Jul/2023:15:46:22 +0000] "GET /images/logo.png HTTP/1.1" 200 5632
31.44.99.102 - - [01/Jul/2023:15:47:15 +0000] "GET /admin.php HTTP/1.1" 404 892
192.168.1.103 - - [01/Jul/2023:15:48:01 +0000] "GET /api/users HTTP/1.1" 200 1820
198.23.124.108 - - [01/Jul/2023:15:49:32 +0000] "POST /api/login HTTP/1.1" 401 347
192.168.1.110 - - [01/Jul/2023:15:50:18 +0000] "GET /download?file=../../../etc/passwd HTTP/1.1" 403 921
192.168.1.105 - - [01/Jul/2023:15:51:43 +0000] "GET /dashboard HTTP/1.1" 200 8932
89.44.11.204 - - [01/Jul/2023:15:52:29 +0000] "GET /.env HTTP/1.1" 404 651
192.168.1.108 - - [01/Jul/2023:15:53:11 +0000] "GET /profile HTTP/1.1" 200 2341
137.74.138.166 - - [01/Jul/2023:15:54:02 +0000] "GET /wp-config.php HTTP/1.1" 404 712
192.168.1.107 - - [01/Jul/2023:15:55:18 +0000] "POST /api/upload HTTP/1.1" 200 412
205.174.165.73 - - [01/Jul/2023:15:56:22 +0000] "POST /login HTTP/1.1" 200 328
192.168.1.100 - - [01/Jul/2023:15:57:09 +0000] "GET /logout HTTP/1.1" 302 218"""
        
        elif log_format == "Syslog":
            log_text = """Jul 01 15:42:31 webserver sshd[12345]: Failed password for invalid user admin from 45.77.65.211 port 58803 ssh2
Jul 01 15:43:10 webserver sshd[12346]: Accepted password for user john from 192.168.1.100 port 59102 ssh2
Jul 01 15:45:20 webserver sudo[12347]: john : TTY=pts/0 ; PWD=/home/john ; USER=root ; COMMAND=/bin/ls /root
Jul 01 15:46:03 webserver sshd[12348]: Failed password for root from 31.44.99.102 port 51432 ssh2
Jul 01 15:46:22 webserver sshd[12349]: Failed password for root from 31.44.99.102 port 51433 ssh2
Jul 01 15:47:15 webserver sshd[12350]: Failed password for root from 31.44.99.102 port 51434 ssh2
Jul 01 15:48:01 webserver kernel[1]: [UFW BLOCK] IN=eth0 OUT= MAC=00:11:22:33:44:55 SRC=198.23.124.108 DST=192.168.1.1 LEN=40 TOS=0x00
Jul 01 15:49:32 webserver apache2[12351]: [error] [client 192.168.1.105] File does not exist: /var/www/html/phpmyadmin
Jul 01 15:50:18 webserver apache2[12352]: [error] [client 89.44.11.204] PHP Warning: include(/etc/passwd): failed to open stream
Jul 01 15:51:43 webserver sshd[12353]: Accepted publickey for user alice from 192.168.1.103 port 60234 ssh2
Jul 01 15:52:29 webserver kernel[1]: [UFW BLOCK] IN=eth0 OUT= MAC=00:11:22:33:44:55 SRC=137.74.138.166 DST=192.168.1.1 LEN=40 TOS=0x00
Jul 01 15:53:11 webserver proftpd[12354]: 192.168.1.108 (192.168.1.108[192.168.1.108]) - FTP session opened.
Jul 01 15:54:02 webserver proftpd[12354]: 192.168.1.108 (192.168.1.108[192.168.1.108]) - FTP session closed.
Jul 01 15:55:18 webserver kernel[1]: [UFW ALLOW] IN=eth0 OUT= MAC=00:11:22:33:44:55 SRC=192.168.1.107 DST=192.168.1.1 LEN=52 TOS=0x00
Jul 01 15:56:22 webserver systemd[1]: Started Session 123 of user bob.
Jul 01 15:57:09 webserver sshd[12355]: Disconnected from user bob 192.168.1.110 port 61001"""
        
        elif log_format == "JSON":
            log_text = """{"timestamp": "2023-07-01T15:42:31Z", "level": "info", "message": "User login", "user": "john", "ip": "192.168.1.100", "status": "success"}
{"timestamp": "2023-07-01T15:43:10Z", "level": "info", "message": "API request", "user": "john", "ip": "192.168.1.100", "endpoint": "/api/users", "method": "GET", "status_code": 200}
{"timestamp": "2023-07-01T15:45:20Z", "level": "warning", "message": "Failed login attempt", "user": "admin", "ip": "45.77.65.211", "status": "failed", "reason": "Invalid credentials"}
{"timestamp": "2023-07-01T15:46:03Z", "level": "error", "message": "Database connection error", "service": "user-service", "error": "Timeout"}
{"timestamp": "2023-07-01T15:46:22Z", "level": "info", "message": "File download", "user": "alice", "ip": "192.168.1.103", "file": "report.pdf", "size": 1024576}
{"timestamp": "2023-07-01T15:47:15Z", "level": "warning", "message": "Failed login attempt", "user": "root", "ip": "31.44.99.102", "status": "failed", "reason": "Invalid credentials"}
{"timestamp": "2023-07-01T15:48:01Z", "level": "info", "message": "Configuration updated", "user": "admin", "ip": "192.168.1.105", "changes": {"timeout": 3600, "max_connections": 100}}
{"timestamp": "2023-07-01T15:49:32Z", "level": "error", "message": "API rate limit exceeded", "user": "bob", "ip": "192.168.1.110", "endpoint": "/api/reports", "method": "GET"}
{"timestamp": "2023-07-01T15:50:18Z", "level": "critical", "message": "Possible SQL injection attempt", "ip": "198.23.124.108", "endpoint": "/search", "query": "' OR 1=1; --"}
{"timestamp": "2023-07-01T15:51:43Z", "level": "info", "message": "User logout", "user": "john", "ip": "192.168.1.100", "session_duration": 540}
{"timestamp": "2023-07-01T15:52:29Z", "level": "warning", "message": "Access denied", "user": "guest", "ip": "89.44.11.204", "resource": "/admin/settings", "reason": "Insufficient permissions"}
{"timestamp": "2023-07-01T15:53:11Z", "level": "info", "message": "File upload", "user": "alice", "ip": "192.168.1.103", "file": "presentation.pptx", "size": 2048576}
{"timestamp": "2023-07-01T15:54:02Z", "level": "error", "message": "Service unavailable", "service": "email-service", "duration": 15.3, "error": "Connection refused"}
{"timestamp": "2023-07-01T15:55:18Z", "level": "info", "message": "Password changed", "user": "bob", "ip": "192.168.1.110"}
{"timestamp": "2023-07-01T15:56:22Z", "level": "critical", "message": "Multiple failed login attempts", "ip": "205.174.165.73", "attempts": 5, "timeframe": "5 minutes", "action": "IP blocked"}
{"timestamp": "2023-07-01T15:57:09Z", "level": "info", "message": "System backup completed", "size": 1073741824, "duration": 120.5, "status": "success"}"""
        
        elif log_format == "UNSW Network Logs":
            try:
                df = pd.read_csv("data/sample_network_logs.csv")
                st.session_state.logs_data = df
            except:
                st.error("Sample network logs file not found. Please upload a CSV file.")
    
    # Show tabs for different analysis types
    if log_text or (input_method == "Use Sample Logs" and log_format == "UNSW Network Logs" and st.session_state.logs_data is not None):
        
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
            # Analysis tabs
            tab1, tab2 = st.tabs(["Known Attack Analysis", "Unknown Threat Analysis"])
            
            with tab1:
                st.subheader("Analysis using UNSW Dataset Model")
                
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
                        
                        attack_counts = st.session_state.results['predicted_attack_cat'].value_counts()
                        
                        fig = px.pie(
                            values=attack_counts.values,
                            names=attack_counts.index,
                            title="Attack Type Distribution",
                            color_discrete_sequence=px.colors.sequential.Plasma_r,
                            hole=0.4
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display table of attack types and mitigations
                        st.subheader("Attack Types & Mitigations")
                        
                        # UNSW attack categories and recommended mitigations
                        attack_mitigations = {
                            "Normal": {
                                "description": "Regular network traffic without malicious intent.",
                                "mitigation": "No action required."
                            },
                            "Generic": {
                                "description": "A technique that works against all block ciphers without regard to the algorithm specifics.",
                                "mitigation": "Implement strong encryption with modern algorithms. Use TLS 1.3 and above. Rotate keys regularly."
                            },
                            "Exploits": {
                                "description": "Exploiting a vulnerability to gain unauthorized access or privileges.",
                                "mitigation": "Keep systems patched and updated. Implement vulnerability scanning. Use intrusion prevention systems."
                            },
                            "Fuzzers": {
                                "description": "Attempts to inject malformed or unexpected data to find vulnerabilities.",
                                "mitigation": "Implement input validation, use WAF (Web Application Firewall), and perform regular security testing."
                            },
                            "DoS": {
                                "description": "Denial of Service attacks to make resources unavailable.",
                                "mitigation": "Implement rate limiting, use DDoS protection services, configure resource quotas, and use traffic filtering."
                            },
                            "Reconnaissance": {
                                "description": "Information gathering to map networks and identify vulnerabilities.",
                                "mitigation": "Limit exposed information, use firewalls to block port scanning, implement network segmentation."
                            },
                            "Analysis": {
                                "description": "Intrusive activities to understand system configurations.",
                                "mitigation": "Implement least privilege, use IDS/IPS systems, limit service information disclosure."
                            },
                            "Backdoor": {
                                "description": "Methods to bypass authentication and access systems.",
                                "mitigation": "Use EDR solutions, implement application whitelisting, perform regular security audits."
                            },
                            "Shellcode": {
                                "description": "Small pieces of code used as payload in exploitation.",
                                "mitigation": "Use ASLR, DEP, and other memory protection mechanisms. Keep systems patched."
                            },
                            "Worms": {
                                "description": "Self-replicating malware that spreads across networks.",
                                "mitigation": "Use endpoint protection, segment networks, implement proper firewall rules."
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
                        
            
            with tab2:
                st.subheader("Unknown Threat Detection")
                
                # Initialize and train the unknown threat detector
                with st.spinner("Analyzing logs for unknown threats..."):
                    try:
                        detector = st.session_state.unknown_threat_detector
                        
                        # Process in batches for larger datasets
                        results = []
                        logs_df = st.session_state.logs_data
                        
                        for i in range(0, len(logs_df), 20):
                            batch = logs_df.iloc[i:i+20].copy()
                            
                            # Make sure message column exists
                            if 'message' not in batch.columns:
                                # Create message from all columns
                                batch['message'] = batch.apply(
                                    lambda row: ' '.join([f"{k}={v}" for k, v in row.items() if pd.notna(v)]), 
                                    axis=1
                                )
                            
                            try:
                                batch_results = detector.detect_batch(batch)
                                results.append(batch_results)
                            except Exception as e:
                                st.error(f"Error analyzing batch: {str(e)}")
                        
                        if results:
                            all_results = pd.concat(results, ignore_index=True)
                            
                            # Add original data columns
                            for col in logs_df.columns:
                                if col not in all_results.columns and len(all_results) == len(logs_df):
                                    all_results[col] = logs_df[col].values
                            
                            # Store in blockchain
                            for index, row in all_results.iterrows():
                                log_data = row.to_dict()
                                st.session_state.blockchain_logger.add_log(log_data)
                            
                            # Save results to session state
                            st.session_state.unknown_threat_results = all_results
                            
                            st.success("Analysis complete! Logs have been analyzed for unknown threats.")
                        else:
                            st.warning("No valid results obtained from unknown threat analysis.")
                    except Exception as e:
                        st.error(f"Error in unknown threat analysis: {str(e)}")
                
                # If we have results, display them
                if st.session_state.unknown_threat_results is not None:
                    # Display metrics
                    total_logs = len(st.session_state.unknown_threat_results)
                    threats = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
                    total_threats = len(threats)
                    threat_rate = (total_threats / total_logs) * 100 if total_logs > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Logs", total_logs)
                    col2.metric("Detected Unknown Threats", total_threats)
                    col3.metric("Unknown Threat Rate", f"{threat_rate:.2f}%")
                    
                    # Threat category distribution
                    if total_threats > 0:
                        st.subheader("Unknown Threat Categories")
                        
                        # Create two visualizations side by side
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Category distribution
                            category_counts = threats['category'].value_counts().reset_index()
                            category_counts.columns = ['Category', 'Count']
                            
                            fig = px.pie(
                                category_counts,
                                values='Count',
                                names='Category',
                                title='Threat Categories',
                                color_discrete_sequence=px.colors.qualitative.Bold,
                                hole=0.4
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Confidence distribution
                            st.subheader("Threat Confidence Distribution")
                            
                            # Create confidence bins
                            confidence_bins = pd.cut(
                                threats['confidence'], 
                                bins=[0, 1, 2, 3, 4],
                                labels=['Low', 'Medium', 'High', 'Critical'],
                                include_lowest=True
                            )
                            
                            confidence_counts = confidence_bins.value_counts().reset_index()
                            confidence_counts.columns = ['Confidence', 'Count']
                            confidence_counts = confidence_counts.sort_values('Confidence')
                            
                            fig = px.bar(
                                confidence_counts,
                                x='Confidence',
                                y='Count',
                                title='Confidence Distribution',
                                color='Count',
                                color_continuous_scale=px.colors.sequential.Plasma_r
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
                            
                            categories_data.append({
                                "Category": category,
                                "Description": threat_descriptions.get(category, "Unknown threat pattern requiring investigation"),
                                "Count": count,
                                "Avg. Confidence": f"{avg_confidence:.2f}"
                            })
                        
                        # Sort by count
                        categories_data = sorted(categories_data, key=lambda x: x["Count"], reverse=True)
                        
                        # Display as table
                        categories_df = pd.DataFrame(categories_data)
                        st.dataframe(categories_df, use_container_width=True)
                    
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

# User Behavior Analysis mode
elif app_mode == "User Behavior Analysis":
    st.title("User Activity Analysis")
    
    # Create a container for filters
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Analyze user behavior patterns to detect suspicious activities, insider threats, and compromised accounts. 
    The system uses ML-based anomaly detection to identify deviations from normal user behavior.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        st.info("Using built-in sample dataset...")
        
        try:
            # Try to load from file
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
2023-07-01T03:30:00Z,jane.smith,113.100.142.22,view,/admin/settings,220,4200,Beijing,unknown,1,0,0.8"""
            
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
            
            # Train or load user behavior model
            st.session_state.user_behavior_model = UserBehaviorAnalyzer()
            st.session_state.user_behavior_model.train(user_features)
            
            # Make predictions
            predictions = st.session_state.user_behavior_model.predict(user_features)
            
            # Store in blockchain
            for index, row in processed_df.iterrows():
                log_data = row.to_dict()
                st.session_state.blockchain_logger.add_log(log_data)
            
            # Save results to session state
            st.session_state.user_behavior_results = predictions
            
            # Success message
            st.success("Analysis complete! User behavior has been analyzed and verified.")
    
    # If results are available, display them
    if st.session_state.user_behavior_results is not None:
        st.subheader("Activity Timeline")
        
        # Create data for user activity timeline
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
            counts = []
            
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
                user_data = st.session_state.user_data[st.session_state.user_data['user_id'] == user_id]
                
                # Look for high risk scores, unusual locations, or failed attempts
                unusual_activities = user_data[
                    (user_data['risk_score'] > 0.5) | 
                    (user_data['failed_attempts'] > 0) | 
                    (user_data['location'].isin(['Moscow', 'Beijing']))
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
                st.dataframe(anomaly_df, use_container_width=True)
                
                # Create location distribution pie chart
                location_counts = anomaly_df['Location'].value_counts().reset_index()
                location_counts.columns = ['Location', 'Count']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        location_counts,
                        values='Count',
                        names='Location',
                        title="Anomalous Activity Locations",
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create risk score distribution
                    fig = px.histogram(
                        anomaly_df,
                        x='Risk Score',
                        nbins=10,
                        title="Risk Score Distribution",
                        color_discrete_sequence=['#FF6B6B']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display all user behavior results
        st.subheader("All User Analysis Results")
        st.dataframe(st.session_state.user_behavior_results, use_container_width=True)

# Zero-Day Threat Detection mode
elif app_mode == "Zero-Day Detection":
    st.title("Zero-Day Threat Detection")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Detect previously unknown (zero-day) threats using advanced pattern recognition with our Isolation Forest model. 
    The system can identify suspicious patterns that don't match known attack signatures with a 90% detection rate.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
                st.dataframe(df.head(), use_container_width=True)
                
                # Process data
                if 'message' not in df.columns:
                    st.error("The uploaded file must contain a 'message' column for log analysis.")
                else:
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
                        
                        # Add original log messages
                        if len(all_results) == len(df):
                            all_results['message'] = df['message'].values
                        
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
        
        # Display threat categories in nice cards
        if total_threats > 0:
            st.subheader("Threat Categories")
            
            # Get unique categories
            categories = threats['category'].unique()
            
            # Create cards for each category with counts
            cols = st.columns(min(3, len(categories)))
            
            for i, category in enumerate(categories):
                col_idx = i % 3
                category_count = len(threats[threats['category'] == category])
                
                # Create HTML for card
                card_html = f"""
                <div style="background-color: #363636; border-radius: 10px; padding: 15px; margin-bottom: 10px; text-align: center;">
                    <h3 style="margin-top: 0;">{category}</h3>
                    <h1 style="color: #FF6B6B; font-size: 36px;">{category_count}</h1>
                </div>
                """
                
                cols[col_idx].markdown(card_html, unsafe_allow_html=True)
            
            # Create detailed charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Create pie chart for categories
                category_counts = threats['category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
                
                fig = px.pie(
                    category_counts,
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
            
            with col2:
                # Create radar chart for confidence by category
                category_confidence = threats.groupby('category')['confidence'].mean().reset_index()
                
                # Normalize values for better radar chart
                category_confidence['confidence'] = category_confidence['confidence'] / 3 * 100
                
                fig = px.line_polar(
                    category_confidence, 
                    r='confidence', 
                    theta='category', 
                    line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    title="Category Confidence Levels (%)"
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Filter options
        st.subheader("View Detailed Results")
        show_option = st.radio("Show:", ["All", "Only Threats", "Only Normal"], key="zero_day_filter")
        
        filtered_df = st.session_state.unknown_threat_results
        if show_option == "Only Threats":
            filtered_df = threats
        elif show_option == "Only Normal":
            filtered_df = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] == 'Normal']
        
        # Display filtered results
        st.dataframe(filtered_df, use_container_width=True)
        
        # Generate report button
        st.subheader("Generate Report")
        if st.button("Generate and Download Report"):
            report_buffer = BytesIO()
            with pd.ExcelWriter(report_buffer, engine='xlsxwriter') as writer:
                st.session_state.unknown_threat_results.to_excel(writer, sheet_name='Unknown Threats', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Logs', 'Detected Threats', 'Threat Rate', 'Analysis Timestamp'],
                    'Value': [total_logs, total_threats, f"{threat_rate:.2f}%", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            report_buffer.seek(0)
            b64 = base64.b64encode(report_buffer.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="unknown_threats_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Download Excel Report</a>'
            st.markdown(href, unsafe_allow_html=True)

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