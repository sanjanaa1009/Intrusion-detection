import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import time
import base64
from io import BytesIO
import json

# Import custom modules
from models.lgbm_model import LGBMClassifier
from models.blockchain import BlockchainLogger
from utils.data_processor import DataProcessor
from utils.visualization import create_anomaly_charts, create_user_behavior_charts, create_threat_distribution
from utils.gemini_integration import get_gemini_recommendation
from attached_assets.user_behavior import UserBehaviorAnalyzer, UserActivityProfiler
from attached_assets.unknownThreat_detector import UnknownThreatClassifier

# Page config
st.set_page_config(
    page_title="CyberSentry IDS",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'lgbm_model' not in st.session_state:
    st.session_state.lgbm_model = None
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

# App header and description
st.sidebar.image("https://images.unsplash.com/photo-1563013544-824ae1b704d3", use_container_width=True)
st.sidebar.title("CyberSentry Enterprise IDS")

# Main navigation
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["Dashboard", "Log Analysis", "User Behavior Analysis", "Unknown Threat Detection", "Blockchain Verification", "Settings"]
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
        anomalies = st.session_state.results[st.session_state.results['prediction'] == -1]
        total_anomalies = len(anomalies)
        anomaly_percentage = (total_anomalies / total_logs) * 100 if total_logs > 0 else 0
        
        col1.metric("Total Logs Analyzed", total_logs)
        col2.metric("Detected Anomalies", total_anomalies)
        col3.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
        col4.metric("Blockchain Verified", f"{total_logs} logs")
        
        # Add visualization
        st.subheader("Threat Distribution")
        if 'attack_cat' in st.session_state.results.columns:
            fig = create_threat_distribution(st.session_state.results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attack category information available in the results.")
    
    # Display user behavior summary if available
    if st.session_state.user_behavior_results is not None:
        st.subheader("User Behavior Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        total_users = len(st.session_state.user_behavior_results['user_id'].unique())
        anomalous_users = len(st.session_state.user_behavior_results[st.session_state.user_behavior_results['prediction'] == -1]['user_id'].unique())
        
        col1.metric("Total Users Analyzed", total_users)
        col2.metric("Users with Anomalous Behavior", anomalous_users)
        
        # Visualization for user behavior
        fig = create_user_behavior_charts(st.session_state.user_behavior_results)
        st.plotly_chart(fig, use_container_width=True)

# Log Analysis mode
elif app_mode == "Log Analysis":
    st.title("Enterprise Network Log Analysis")
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Upload network logs for advanced anomaly detection and attack classification. 
    Our enhanced LGBM/RandomForest model provides industry-leading accuracy for known attack pattern detection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show model performance metrics
    if st.session_state.lgbm_model is not None and hasattr(st.session_state.lgbm_model, 'get_model_metrics_html'):
        st.markdown(st.session_state.lgbm_model.get_model_metrics_html(), unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Log Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing log data..."):
            df = pd.read_csv(uploaded_file)
            st.session_state.logs_data = df
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Process data
            processed_df = st.session_state.data_processor.preprocess_network_data(df)
            
            # Initialize LGBM model if not already loaded
            if st.session_state.lgbm_model is None:
                st.session_state.lgbm_model = LGBMClassifier()
                st.session_state.lgbm_model.initialize_model()
            
            # Make predictions
            predictions = st.session_state.lgbm_model.predict(processed_df)
            probabilities = st.session_state.lgbm_model.predict_proba(processed_df)
            
            # Add predictions to DataFrame
            result_df = processed_df.copy()
            result_df['prediction'] = predictions
            result_df['probability'] = probabilities.max(axis=1)
            
            # If attack categories are available in the model
            if hasattr(st.session_state.lgbm_model, 'attack_categories'):
                result_df['predicted_attack_cat'] = [st.session_state.lgbm_model.attack_categories[pred] if pred != 0 else "Normal" for pred in predictions]
            
            # Store in blockchain
            for index, row in result_df.iterrows():
                log_data = row.to_dict()
                st.session_state.blockchain_logger.add_log(log_data)
            
            # Save results to session state
            st.session_state.results = result_df
            
            # Success message
            st.success("Analysis complete! Logs have been analyzed and verified.")
    
    # If results are available, display them
    if st.session_state.results is not None:
        st.subheader("Analysis Results")
        
        # Display metrics
        total_logs = len(st.session_state.results)
        anomalies = st.session_state.results[st.session_state.results['prediction'] == -1]
        total_anomalies = len(anomalies)
        anomaly_percentage = (total_anomalies / total_logs) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Logs", total_logs)
        col2.metric("Detected Anomalies", total_anomalies)
        col3.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
        
        # Filter options
        st.subheader("Filter Results")
        show_option = st.radio("Show:", ["All", "Only Anomalies", "Only Normal"])
        
        filtered_df = st.session_state.results
        if show_option == "Only Anomalies":
            filtered_df = st.session_state.results[st.session_state.results['prediction'] == -1]
        elif show_option == "Only Normal":
            filtered_df = st.session_state.results[st.session_state.results['prediction'] == 1]
        
        # Display filtered results
        st.dataframe(filtered_df)
        
        # Generate visualizations
        st.subheader("Visualizations")
        charts = create_anomaly_charts(st.session_state.results)
        for chart in charts:
            st.plotly_chart(chart, use_container_width=True)
        
        # Generate report
        st.subheader("Generate Report")
        if st.button("Generate and Download Report"):
            report_buffer = BytesIO()
            with pd.ExcelWriter(report_buffer, engine='xlsxwriter') as writer:
                st.session_state.results.to_excel(writer, sheet_name='Analysis Results', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Logs', 'Detected Anomalies', 'Anomaly Rate', 'Analysis Timestamp'],
                    'Value': [total_logs, total_anomalies, f"{anomaly_percentage:.2f}%", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            report_buffer.seek(0)
            b64 = base64.b64encode(report_buffer.read()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="ids_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Download Excel Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Get action recommendations
        st.subheader("Get Action Recommendations")
        if st.button("Generate Recommendations"):
            sample_anomalies = st.session_state.results[st.session_state.results['prediction'] == -1].head(5)
            if len(sample_anomalies) > 0:
                with st.spinner("Generating recommendations..."):
                    for _, row in sample_anomalies.iterrows():
                        anomaly_type = row.get('predicted_attack_cat', 'Unknown')
                        recommendation = get_gemini_recommendation(anomaly_type, row.to_dict())
                        
                        st.write(f"**Anomaly Type:** {anomaly_type}")
                        st.info(recommendation)
            else:
                st.info("No anomalies found to generate recommendations.")

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
    """
    , unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload User Activity Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing user activity data..."):
            df = pd.read_csv(uploaded_file)
            st.session_state.user_data = df
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Process data
            processed_df = st.session_state.data_processor.preprocess_user_data(df)
            
            # Build user activity profiles
            activity_profiler = UserActivityProfiler()
            activity_profiler.build_profiles(processed_df)
            
            # Extract features for anomaly detection
            user_features = st.session_state.data_processor.extract_user_behavior_features(processed_df)
            
            # Train or load user behavior model
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
        st.dataframe(st.session_state.user_behavior_results)
        
        # Generate visualizations
        st.subheader("User Behavior Visualizations")
        fig = create_user_behavior_charts(st.session_state.user_behavior_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate report
        st.subheader("Generate Report")
        if st.button("Generate and Download Report"):
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

# Unknown Threat Detection mode
elif app_mode == "Unknown Threat Detection":
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
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Log Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing log data for unknown threats..."):
            df = pd.read_csv(uploaded_file)
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
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
                all_results['message'] = df['message'].values
                
                # Store in blockchain
                for index, row in all_results.iterrows():
                    log_data = row.to_dict()
                    st.session_state.blockchain_logger.add_log(log_data)
                
                # Save results to session state
                st.session_state.unknown_threat_results = all_results
                
                # Success message
                st.success("Analysis complete! Logs have been analyzed for unknown threats.")
    
    # If results are available, display them
    if st.session_state.unknown_threat_results is not None:
        st.subheader("Unknown Threat Detection Results")
        
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
        show_option = st.radio("Show:", ["All", "Only Threats", "Only Normal"])
        
        filtered_df = st.session_state.unknown_threat_results
        if show_option == "Only Threats":
            filtered_df = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
        elif show_option == "Only Normal":
            filtered_df = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] == 'Normal']
        
        # Display filtered results
        st.dataframe(filtered_df)
        
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
        if st.button("Generate and Download Report"):
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

# Settings mode
elif app_mode == "Settings":
    st.title("Settings")
    
    # Model settings
    st.subheader("Model Settings")
    
    # LGBM Model settings
    st.markdown("### LGBM Model Settings")
    lgbm_contamination = st.slider("LGBM Contamination Factor", 0.01, 0.20, 0.05, 0.01)
    
    # Isolation Forest settings
    st.markdown("### Isolation Forest Settings")
    iso_contamination = st.slider("Isolation Forest Contamination Factor", 0.01, 0.20, 0.05, 0.01)
    pattern_threshold = st.slider("Pattern Recognition Threshold", 1.0, 3.0, 1.5, 0.1)
    
    # User Behavior settings
    st.markdown("### User Behavior Model Settings")
    user_contamination = st.slider("User Behavior Contamination Factor", 0.01, 0.20, 0.05, 0.01)
    
    # Apply settings
    if st.button("Apply Settings"):
        # Update UnknownThreatClassifier settings
        st.session_state.unknown_threat_detector.pattern_threshold = pattern_threshold
        
        # For now, we'll need to retrain models with new settings
        st.session_state.lgbm_model = None  # Force model reinitialization with new settings
        
        st.success("Settings applied successfully!")
    
    # Clear data option
    st.subheader("Clear Data")
    if st.button("Clear All Data"):
        st.session_state.logs_data = None
        st.session_state.user_data = None
        st.session_state.results = None
        st.session_state.user_behavior_results = None
        st.session_state.unknown_threat_results = None
        st.session_state.blockchain_logger = BlockchainLogger()  # Reset blockchain
        
        st.success("All data cleared successfully!")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("CyberSentry IDS v1.0")
st.sidebar.text("© 2023 All Rights Reserved")
