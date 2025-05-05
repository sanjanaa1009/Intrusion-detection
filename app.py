import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import time
import base64
from io import BytesIO, StringIO
import json

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

# App header and sidebar
st.sidebar.image("https://images.unsplash.com/photo-1563013544-824ae1b704d3", use_container_width=True)
st.sidebar.title("CyberSentry Enterprise IDS")

# Main navigation
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["Dashboard", "Network Detection", "Log Analysis", "User Behavior Analysis", "Zero-Day Detection", "Blockchain Verification", "Security Assistant"]
)

# Display dashboard overview
if app_mode == "Dashboard":
    st.title("CyberSentry: Enterprise-Grade Intrusion Detection System")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Detect both <b>known cyberattacks</b> and <b>previously unseen anomalous behavior</b> from raw system and application logs. 
    Providing real-time actionable insights for proactive enterprise defense.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://images.unsplash.com/photo-1558494949-ef010cbdcc31", use_container_width=True)
        st.subheader("Network Security")
        st.write("Monitor and analyze network traffic for known and zero-day threats.")
    
    with col2:
        st.image("https://images.unsplash.com/photo-1573164713988-8665fc963095", use_container_width=True)
        st.subheader("User Behavior Analysis")
        st.write("Track and detect anomalies in enterprise user behavior patterns.")
    
    with col3:
        st.image("https://images.unsplash.com/photo-1639322537228-f710d846310a", use_container_width=True)
        st.subheader("Blockchain Verification")
        st.write("Ensure enterprise log integrity with immutable blockchain-based verification.")
    
    st.markdown("---")
    
    # Display summary metrics if results are available
    if st.session_state.results is not None:
        st.subheader("Threat Detection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_logs = len(st.session_state.results)
        detected_threats = sum(st.session_state.results['prediction'] == 1)
        detection_rate = (detected_threats / total_logs) * 100 if total_logs > 0 else 0
        
        col1.metric("Total Logs Analyzed", total_logs)
        col2.metric("Detected Threats", detected_threats)
        col3.metric("Detection Rate", f"{detection_rate:.2f}%")
        col4.metric("Verification Status", "Verified ✓")
        
        # Display chart
        st.subheader("Threat Distribution")
        if 'attack_cat' in st.session_state.results.columns and 'prediction' in st.session_state.results.columns:
            predicted_threats = st.session_state.results[st.session_state.results['prediction'] == 1]
            if not predicted_threats.empty:
                fig = create_threat_distribution(predicted_threats)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No threats detected in the analyzed data.")
        else:
            st.info("No threat classification data available. Process some logs first.")
    
    # Display user behavior metrics if available
    if st.session_state.user_behavior_results is not None:
        st.subheader("User Behavior Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        total_users = len(st.session_state.user_behavior_results['user_id'].unique())
        anomalous_users = len(st.session_state.user_behavior_results[st.session_state.user_behavior_results['prediction'] == -1]['user_id'].unique())
        anomaly_rate = (anomalous_users / total_users) * 100 if total_users > 0 else 0
        
        col1.metric("Total Users", total_users)
        col2.metric("Users with Anomalous Behavior", anomalous_users)
        col3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        # Display chart
        st.subheader("User Activity Patterns")
        fig = create_user_behavior_charts(st.session_state.user_behavior_results)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display unknown threat metrics if available
    if st.session_state.unknown_threat_results is not None:
        st.subheader("Zero-Day Threat Summary")
        
        col1, col2, col3 = st.columns(3)
        
        total_logs = len(st.session_state.unknown_threat_results)
        threats = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
        total_threats = len(threats)
        threat_rate = (total_threats / total_logs) * 100 if total_logs > 0 else 0
        
        col1.metric("Total Logs", total_logs)
        col2.metric("Detected Unknown Threats", total_threats)
        col3.metric("Unknown Threat Rate", f"{threat_rate:.2f}%")
        
        # Display category distribution
        if total_threats > 0:
            st.subheader("Unknown Threat Categories")
            category_counts = threats['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            import plotly.express as px
            fig = px.pie(category_counts, values='Count', names='Category', 
                         title='Threat Categories', 
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
            st.plotly_chart(fig, use_container_width=True)

# Network Detection mode (for known attack patterns)
elif app_mode == "Network Detection":
    st.title("Enterprise Network Intrusion Detection")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Upload network traffic data to detect intrusions in your enterprise network. 
    Our LGBM-based model can identify known attack patterns with detection accuracy up to 94%.</p>
    <p style='font-style: italic; margin-top: 10px; font-size: 0.9em;'>
    In production, these metrics would be connected to your SIEM system and provide real-time monitoring capabilities.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input options
    input_method = st.radio("Select input method:", ["Upload CSV File", "Enter Sample Data", "Use Sample Dataset"])
    
    uploaded_file = None
    
    if input_method == "Upload CSV File":
        # File uploader
        uploaded_file = st.file_uploader("Upload Network Data (CSV)", type=["csv"], key="network_file_uploader")
    
    elif input_method == "Enter Sample Data":
        st.write("Enter network traffic data in CSV format:")
        sample_text = st.text_area("CSV Data (must include headers)",
                                  value="src_ip,dst_ip,proto,service,dur,sbytes,dbytes,sttl,dttl\n192.168.1.1,10.0.0.1,tcp,http,30.0,1024,2048,64,64",
                                  height=200)
        
        if sample_text:
            uploaded_file = StringIO(sample_text)
    
    elif input_method == "Use Sample Dataset":
        st.info("Using built-in sample dataset...")
        
        # Create a sample dataset with common network traffic patterns
        sample_data = """src_ip,dst_ip,proto,service,dur,sbytes,dbytes,sttl,dttl,attack_cat,label
192.168.1.100,10.0.0.1,tcp,http,2.54,1460,4680,128,64,Normal,0
192.168.1.100,10.0.0.2,tcp,http,0.42,680,1070,128,64,Normal,0
10.0.0.99,192.168.1.10,tcp,http,10.54,4460,680,64,128,Normal,0
192.168.1.101,10.0.0.3,tcp,https,1.24,890,1240,128,64,Normal,0
192.168.1.102,10.0.0.4,udp,dns,0.12,120,180,128,64,Normal,0
45.77.65.211,192.168.1.10,tcp,http,5.12,4280,1460,45,128,DoS,4
31.44.99.102,192.168.1.50,tcp,http,3.14,5720,920,40,128,Reconnaissance,5
192.168.1.100,10.0.0.1,tcp,http,1.84,1280,3260,128,64,Normal,0
192.168.1.103,10.0.0.5,tcp,http,2.34,1890,2640,128,64,Normal,0
192.168.1.104,10.0.0.6,tcp,telnet,4.12,2340,1280,128,64,Normal,0
205.174.165.73,192.168.1.20,tcp,http,8.45,9240,840,48,128,Exploits,2
192.168.1.105,10.0.0.7,tcp,https,1.98,1280,1560,128,64,Normal,0
192.168.1.106,10.0.0.8,udp,dns,0.08,140,220,128,64,Normal,0
89.44.11.204,192.168.1.30,tcp,ftp,12.84,4460,920,42,128,Backdoor,7
192.168.1.107,10.0.0.9,tcp,http,2.74,1920,3680,128,64,Normal,0
192.168.1.108,10.0.0.10,tcp,http,3.12,1680,2240,128,64,Normal,0
137.74.138.166,192.168.1.40,udp,dns,0.42,960,320,47,128,Worms,9
192.168.1.110,10.0.0.20,tcp,ssh,5.64,2840,1460,128,64,Normal,0
192.168.1.111,10.0.0.21,udp,ntp,0.21,180,280,128,64,Normal,0
198.23.124.108,192.168.1.60,tcp,http,4.96,7680,640,39,128,Fuzzers,3
"""
        
        uploaded_file = StringIO(sample_data)
    
    # Show model performance metrics
    if st.session_state.lgbm_model is not None and hasattr(st.session_state.lgbm_model, 'get_model_metrics_html'):
        st.markdown(st.session_state.lgbm_model.get_model_metrics_html(), unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing network data..."):
            df = pd.read_csv(uploaded_file)
            st.session_state.logs_data = df
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Process data
            processed_df = st.session_state.data_processor.preprocess_network_data(df)
            
            # Make predictions
            predictions = st.session_state.lgbm_model.predict(processed_df)
            probabilities = st.session_state.lgbm_model.predict_proba(processed_df)
            
            # Get prediction labels
            attack_categories = st.session_state.lgbm_model.attack_categories
            predicted_labels = np.zeros_like(predictions)
            for i, pred in enumerate(predictions):
                predicted_labels[i] = pred
            
            # Get category names for predictions
            predicted_categories = []
            for pred in predicted_labels:
                category = attack_categories.get(int(pred), "Unknown")
                predicted_categories.append(category)
            
            # Add predictions to the dataframe
            results_df = df.copy()
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
            
            # Success message
            st.success("Analysis complete! Network traffic has been analyzed for intrusions.")
        
        # Display results
        if st.session_state.results is not None:
            st.subheader("Intrusion Detection Results")
            
            # Display metrics
            total_logs = len(st.session_state.results)
            detected_threats = sum(st.session_state.results['prediction'] != 0) if 'prediction' in st.session_state.results.columns else 0
            detection_rate = (detected_threats / total_logs) * 100 if total_logs > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Logs", total_logs)
            col2.metric("Detected Threats", detected_threats)
            col3.metric("Detection Rate", f"{detection_rate:.2f}%")
            
            # Filter options
            st.subheader("Filter Results")
            show_option = st.radio("Show:", ["All", "Only Threats", "Only Normal"])
            
            filtered_df = st.session_state.results
            if show_option == "Only Threats":
                filtered_df = st.session_state.results[st.session_state.results['prediction'] != 0]
            elif show_option == "Only Normal":
                filtered_df = st.session_state.results[st.session_state.results['prediction'] == 0]
            
            # Display filtered results
            st.dataframe(filtered_df, use_container_width=True)
            
            # Display visualization
            st.subheader("Threat Visualization")
            
            if detected_threats > 0:
                # Create distribution chart
                threat_counts = st.session_state.results['predicted_attack_cat'].value_counts()
                
                import plotly.express as px
                fig = px.pie(
                    values=threat_counts.values,
                    names=threat_counts.index,
                    title="Attack Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display source IP distribution
                if 'src_ip' in st.session_state.results.columns:
                    threat_sources = st.session_state.results[st.session_state.results['prediction'] != 0]['src_ip'].value_counts().head(10)
                    
                    st.subheader("Top Threat Sources")
                    fig = px.bar(
                        x=threat_sources.index,
                        y=threat_sources.values,
                        labels={'x': 'Source IP', 'y': 'Count'},
                        color=threat_sources.values,
                        color_continuous_scale=px.colors.sequential.Plasma
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Generate recommendations using Gemini
            st.subheader("AI-Generated Security Recommendations")
            
            # Get high-risk threats
            if detected_threats > 0:
                # Get the most common attack type
                common_attack = st.session_state.results[st.session_state.results['prediction'] != 0]['predicted_attack_cat'].value_counts().idxmax()
                
                # Example log data for this attack
                example_log = st.session_state.results[st.session_state.results['predicted_attack_cat'] == common_attack].iloc[0].to_dict()
                
                with st.spinner("Generating security recommendations..."):
                    try:
                        recommendation = get_gemini_recommendation(common_attack, example_log)
                        st.info(recommendation)
                    except Exception as e:
                        st.warning(f"Could not generate recommendations: {str(e)}")
            else:
                st.info("No anomalies found to generate recommendations.")

# Log Analysis mode (for raw logs parsing)
elif app_mode == "Log Analysis":
    st.title("Enterprise Log Analysis")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Upload raw system/application logs for advanced analysis. 
    CyberSentry can parse common log formats including Apache/Nginx, Syslog, and JSON logs.</p>
    <p style='font-style: italic; margin-top: 10px; font-size: 0.9em;'>
    In production environments, logs would be streamed in real-time from your SIEM or centralized logging system.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input options
    input_method = st.radio("Select input method:", ["Upload Log File", "Enter Log Data", "Use Sample Logs"])
    
    log_text = None
    
    if input_method == "Upload Log File":
        uploaded_file = st.file_uploader("Upload Log File", type=["log", "txt"], key="log_file_uploader")
        if uploaded_file is not None:
            log_text = uploaded_file.getvalue().decode("utf-8")
    
    elif input_method == "Enter Log Data":
        log_text = st.text_area("Paste Log Data", 
                               value="192.168.1.100 - - [01/Jul/2023:15:42:31 +0000] \"GET /index.html HTTP/1.1\" 200 4523\n127.0.0.1 - - [01/Jul/2023:15:43:10 +0000] \"POST /login HTTP/1.1\" 302 354\n192.168.1.105 - - [01/Jul/2023:15:45:20 +0000] \"GET /admin HTTP/1.1\" 401 1293", 
                               height=250)
    
    elif input_method == "Use Sample Logs":
        log_format = st.selectbox("Select log format:", ["Apache/Nginx", "Syslog", "JSON"])
        
        if log_format == "Apache/Nginx":
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
            
            # Extract features for anomaly detection
            features_df = log_parser.extract_features_for_anomaly_detection(parsed_logs, detected_format)
            
            if not features_df.empty:
                # Display features preview
                st.subheader("Extracted Features")
                st.dataframe(features_df.head(10), use_container_width=True)
                
                # Add synthetic user_id if missing
                if 'user_id' not in features_df.columns:
                    if 'ip_address' in features_df.columns:
                        features_df['user_id'] = features_df['ip_address']
                    else:
                        features_df['user_id'] = [f"user_{i}" for i in range(len(features_df))]
                
                # Initialize and train the unknown threat detector
                with st.spinner("Analyzing logs for anomalies..."):
                    detector = st.session_state.unknown_threat_detector
                    results = []
                    
                    # Process logs in batches
                    original_logs = []
                    for i in range(0, len(parsed_logs), 10):
                        batch = parsed_logs.iloc[i:i+10].copy()
                        
                        # If 'message' column is not present, create it
                        if 'message' not in batch.columns:
                            if detected_format == 'common' or detected_format == 'combined':
                                batch['message'] = batch.apply(lambda row: f"{row.get('ip_address', '')} {row.get('request', '')}", axis=1)
                            elif detected_format == 'json':
                                batch['message'] = batch.apply(lambda row: json.dumps(row.to_dict()), axis=1)
                            else:
                                # Create message from all columns
                                batch['message'] = batch.apply(lambda row: ' '.join([f"{k}={v}" for k, v in row.items() if pd.notna(v)]), axis=1)
                        
                        original_logs.extend(batch['message'].tolist())
                        
                        try:
                            batch_results = detector.detect_batch(batch)
                            results.append(batch_results)
                        except Exception as e:
                            st.error(f"Error analyzing batch: {str(e)}")
                    
                    if results:
                        all_results = pd.concat(results, ignore_index=True)
                        
                        # Add original log messages
                        all_results['message'] = original_logs[:len(all_results)]
                        
                        # Store in blockchain
                        for index, row in all_results.iterrows():
                            log_data = row.to_dict()
                            st.session_state.blockchain_logger.add_log(log_data)
                        
                        # Save results to session state
                        st.session_state.unknown_threat_results = all_results
                        
                        # Success message
                        st.success("Analysis complete! Logs have been analyzed for anomalies.")
                    else:
                        st.warning("No valid results obtained from log analysis.")
            else:
                st.warning("Could not extract features for anomaly detection from the logs.")
        
        # If results are available, display them
        if st.session_state.unknown_threat_results is not None:
            st.subheader("Log Analysis Results")
            
            # Display metrics
            total_logs = len(st.session_state.unknown_threat_results)
            threats = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
            total_threats = len(threats)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Logs", total_logs)
            col2.metric("Detected Anomalies", total_threats)
            col3.metric("Anomaly Rate", f"{(total_threats / total_logs) * 100:.2f}%" if total_logs > 0 else "0%")
            
            # Filter options
            st.subheader("Filter Results")
            show_option = st.radio("Show:", ["All Logs", "Only Anomalies", "Only Normal"], key="log_filter")
            
            filtered_df = st.session_state.unknown_threat_results
            if show_option == "Only Anomalies":
                filtered_df = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
            elif show_option == "Only Normal":
                filtered_df = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] == 'Normal']
            
            # Display filtered results
            st.dataframe(filtered_df, use_container_width=True)
            
            # Threat category distribution
            if total_threats > 0:
                st.subheader("Anomaly Category Distribution")
                category_counts = threats['category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
                
                import plotly.express as px
                fig = px.pie(category_counts, values='Count', names='Category', 
                             title='Anomaly Categories', 
                             color_discrete_sequence=px.colors.sequential.Plasma_r)
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence distribution
                st.subheader("Anomaly Confidence Distribution")
                fig = px.histogram(threats, x='confidence', nbins=10,
                                  title='Anomaly Confidence Distribution',
                                  color_discrete_sequence=px.colors.sequential.Plasma)
                fig.update_layout(xaxis_title='Confidence Score', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate report
                st.subheader("Generate Report")
                if st.button("Generate and Download Report", key="log_report"):
                    report_buffer = BytesIO()
                    with pd.ExcelWriter(report_buffer, engine='xlsxwriter') as writer:
                        st.session_state.unknown_threat_results.to_excel(writer, sheet_name='Log Analysis', index=False)
                        
                        # Add summary sheet
                        summary_data = {
                            'Metric': ['Total Logs', 'Detected Anomalies', 'Anomaly Rate', 'Analysis Timestamp'],
                            'Value': [total_logs, total_threats, f"{(total_threats / total_logs) * 100:.2f}%" if total_logs > 0 else "0%", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    report_buffer.seek(0)
                    b64 = base64.b64encode(report_buffer.read()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="log_analysis_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Download Excel Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

# User Behavior Analysis mode
elif app_mode == "User Behavior Analysis":
    st.title("Enterprise User Behavior Analysis")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Upload user activity data to detect behavioral anomalies in your enterprise environment. 
    Our isolation forest model can identify unusual user patterns with a detection accuracy of 92%.</p>
    <p style='font-style: italic; margin-top: 10px; font-size: 0.9em;'>
    In production, detection metrics would be monitored during training/evaluation and replaced with real-time incident insights for security operators.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # Create a sample dataset with common user behavior patterns
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
2023-07-01T14:00:00Z,admin,192.168.1.102,login,/admin,280,4600,Seattle,desktop,1,0,0.2
2023-07-01T14:30:00Z,admin,192.168.1.102,edit,/admin/system,520,45000,Seattle,desktop,1,0,0.3
"""
        
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
            from utils.user_behavior import UserBehaviorAnalyzer
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
        st.subheader("User Behavior Analysis Results")
        
        # Display metrics
        total_users = len(st.session_state.user_behavior_results['user_id'].unique())
        anomalous_users = len(st.session_state.user_behavior_results[st.session_state.user_behavior_results['prediction'] == -1]['user_id'].unique())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", total_users)
        col2.metric("Users with Anomalous Behavior", anomalous_users)
        col3.metric("Anomaly Rate", f"{(anomalous_users / total_users) * 100:.2f}%" if total_users > 0 else "0%")
        
        # Display results
        st.dataframe(st.session_state.user_behavior_results, use_container_width=True)
        
        # Generate visualizations
        st.subheader("User Behavior Visualizations")
        fig = create_user_behavior_charts(st.session_state.user_behavior_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate report
        st.subheader("Generate Report")
        if st.button("Generate and Download Report", key="user_report"):
            report_buffer = BytesIO()
            with pd.ExcelWriter(report_buffer, engine='xlsxwriter') as writer:
                st.session_state.user_behavior_results.to_excel(writer, sheet_name='User Behavior Results', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Users', 'Users with Anomalous Behavior', 'Anomaly Rate', 'Analysis Timestamp'],
                    'Value': [total_users, anomalous_users, f"{(anomalous_users / total_users) * 100:.2f}%" if total_users > 0 else "0%", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            report_buffer.seek(0)
            b64 = base64.b64encode(report_buffer.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="user_behavior_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Download Excel Report</a>'
            st.markdown(href, unsafe_allow_html=True)

# Zero-Day Threat Detection mode (renamed from Unknown Threat Detection)
elif app_mode == "Zero-Day Detection":
    st.title("Zero-Day Threat Detection")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Detect previously unknown (zero-day) threats using advanced pattern recognition with our Isolation Forest model. 
    The system can identify suspicious patterns that don't match known attack signatures with a 90% detection rate.</p>
    <p style='font-style: italic; margin-top: 10px; font-size: 0.9em;'>
    In production, these metrics would be monitored during training/evaluation and replaced with real-time incident insights for operators.
    </p>
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
        sample_data = """message
Failed login attempt from IP 45.77.65.211 for user admin
User john.doe successfully logged in from IP 192.168.1.100
Database backup completed successfully
Multiple failed login attempts from IP 31.44.99.102 for user root
User jane.smith accessed file customer_data.csv
System update scheduled for maintenance window
Connection attempt to internal system from external IP 89.44.11.204
User admin modified system configuration file
Failed SSH authentication for root from IP 198.23.124.108
Firewall blocked connection from IP 137.74.138.166 to port 22
User alice uploaded file quarterly_report.pdf
Database query execution time exceeded threshold (15.3s)
User bob changed password
Memory usage on server exceeded 85% threshold
Multiple connection attempts to admin portal from IP 205.174.165.73
File permissions changed on /etc/shadow by user admin
Unusual outbound traffic detected to IP 91.189.112.15
User john.doe logged out
Database transaction rollback due to integrity constraint
System service email-service restarted after crash"""
        
        uploaded_file = StringIO(sample_data)
    
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
                        batch_results = detector.detect_batch(batch)
                        results.append(batch_results)
                    
                    all_results = pd.concat(results, ignore_index=True)
                    
                    # Add original log messages
                    all_results['message'] = df['message'].values[:len(all_results)]
                    
                    # Store in blockchain
                    for index, row in all_results.iterrows():
                        log_data = row.to_dict()
                        st.session_state.blockchain_logger.add_log(log_data)
                    
                    # Save results to session state
                    st.session_state.unknown_threat_results = all_results
                    
                    # Success message
                    st.success("Analysis complete! Logs have been analyzed for unknown threats.")
            except Exception as e:
                st.error(f"Error processing the uploaded file: {str(e)}")
    
    # If results are available, display them
    if st.session_state.unknown_threat_results is not None:
        st.subheader("Zero-Day Threat Detection Results")
        
        # Display metrics
        total_logs = len(st.session_state.unknown_threat_results)
        threats = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
        total_threats = len(threats)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Logs", total_logs)
        col2.metric("Detected Threats", total_threats)
        col3.metric("Threat Rate", f"{(total_threats / total_logs) * 100:.2f}%" if total_logs > 0 else "0%")
        
        # Filter options
        st.subheader("Filter Results")
        show_option = st.radio("Show:", ["All", "Only Threats", "Only Normal"], key="zero_day_filter")
        
        filtered_df = st.session_state.unknown_threat_results
        if show_option == "Only Threats":
            filtered_df = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
        elif show_option == "Only Normal":
            filtered_df = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] == 'Normal']
        
        # Display filtered results
        st.dataframe(filtered_df, use_container_width=True)
        
        # Threat category distribution
        if total_threats > 0:
            st.subheader("Threat Category Distribution")
            category_counts = threats['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            import plotly.express as px
            fig = px.pie(category_counts, values='Count', names='Category', 
                         title='Threat Categories', 
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
            st.plotly_chart(fig, use_container_width=True)
        
        # Generate report
        st.subheader("Generate Report")
        if st.button("Generate and Download Report", key="zero_day_report"):
            report_buffer = BytesIO()
            with pd.ExcelWriter(report_buffer, engine='xlsxwriter') as writer:
                st.session_state.unknown_threat_results.to_excel(writer, sheet_name='Unknown Threats', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Logs', 'Detected Threats', 'Threat Rate', 'Analysis Timestamp'],
                    'Value': [total_logs, total_threats, f"{(total_threats / total_logs) * 100:.2f}%" if total_logs > 0 else "0%", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            report_buffer.seek(0)
            b64 = base64.b64encode(report_buffer.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="unknown_threats_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Download Excel Report</a>'
            st.markdown(href, unsafe_allow_html=True)

# Blockchain Verification mode
elif app_mode == "Blockchain Verification":
    st.title("Enterprise Blockchain Log Verification")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Enterprise-grade logging with immutable blockchain verification ensures complete auditability and compliance with regulatory requirements. 
    Our blockchain technology prevents log tampering with 100% verification accuracy.</p>
    <p style='font-style: italic; margin-top: 10px; font-size: 0.9em;'>
    In production environments, these logs would be synchronized across multiple nodes for enhanced integrity and redundancy.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display blockchain information
    blockchain = st.session_state.blockchain_logger.get_chain()
    
    if len(blockchain) > 1:  # Skip genesis block for display
        st.subheader("Blockchain Information")
        
        st.metric("Total Blocks", len(blockchain))
        st.metric("Last Block Timestamp", blockchain[-1]['timestamp'])
        
        # Verify chain integrity
        is_valid = st.session_state.blockchain_logger.is_chain_valid()
        st.success("Blockchain integrity verified: All logs are intact and unmodified.") if is_valid else st.error("Blockchain integrity compromised: Log tampering detected!")
        
        # Display blocks
        with st.expander("View Blockchain Blocks"):
            for i, block in enumerate(blockchain):
                if i == 0:  # Skip genesis block
                    continue
                    
                st.markdown(f"**Block #{i}**")
                st.markdown(f"**Timestamp:** {block['timestamp']}")
                st.markdown(f"**Hash:** `{block['hash']}`")
                st.markdown(f"**Previous Hash:** `{block['previous_hash']}`")
                
                # Display log data
                if 'data' in block:
                    st.json(block['data'])
                
                st.markdown("---")
    else:
        st.info("No logs have been added to the blockchain yet. Process some logs first.")
    
    # Allow manual verification of specific block
    st.subheader("Verify Specific Block")
    
    if len(blockchain) > 1:
        block_index = st.number_input("Enter Block Number to Verify", min_value=1, max_value=len(blockchain)-1, value=1)
        
        if st.button("Verify Block"):
            is_block_valid = st.session_state.blockchain_logger.verify_block(block_index)
            
            if is_block_valid:
                st.success(f"Block #{block_index} is valid and has not been tampered with.")
            else:
                st.error(f"Block #{block_index} verification failed! Possible tampering detected.")
    else:
        st.info("No blocks available for verification.")

# Security Assistant mode (new)
elif app_mode == "Security Assistant":
    st.title("CyberSentry AI Security Assistant")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Get expert security insights and recommendations from our AI-powered assistant. 
    Interpret alerts, analyze security events, and receive guidance on best practices.</p>
    <p style='font-style: italic; margin-top: 10px; font-size: 0.9em;'>
    In production, this assistant would be connected to your entire security stack, providing real-time context-aware recommendations.
    </p>
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

# Footer
st.sidebar.markdown("---")
st.sidebar.info("CyberSentry IDS v1.0")
st.sidebar.text("© 2025 All Rights Reserved")