import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_threat_distribution(threats_df):
    """
    Create a visualization of threat distribution by attack category
    
    Parameters:
    -----------
    threats_df : pd.DataFrame
        DataFrame containing detected threats with 'attack_cat' column
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Group by attack category
    if 'attack_cat' in threats_df.columns:
        category_col = 'attack_cat'
    elif 'predicted_attack_cat' in threats_df.columns:
        category_col = 'predicted_attack_cat'
    else:
        # Fall back to other columns that might contain categories
        for possible_col in ['category', 'type', 'classification']:
            if possible_col in threats_df.columns:
                category_col = possible_col
                break
        else:
            # If no suitable column is found, return empty figure
            return go.Figure()
    
    category_counts = threats_df[category_col].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    # Create pie chart
    fig = px.pie(
        category_counts,
        values='Count',
        names='Category',
        title='Threat Distribution by Category',
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.4
    )
    
    # Add source IP information if available
    if 'src_ip' in threats_df.columns:
        top_ips = threats_df['src_ip'].value_counts().head(5).reset_index()
        top_ips.columns = ['Source IP', 'Count']
        
        fig_ips = px.bar(
            top_ips,
            x='Source IP',
            y='Count',
            title='Top 5 Source IPs',
            color='Count',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        
        # Update layout
        fig.update_layout(
            title_text='Threat Analysis',
            annotations=[dict(text='Categories', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig_ips
    
    return fig

def create_user_behavior_charts(user_df):
    """
    Create visualizations for user behavior analysis
    
    Parameters:
    -----------
    user_df : pd.DataFrame
        DataFrame containing user behavior data with 'user_id' and 'prediction' columns
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('Anomaly Distribution', 'Top Users by Anomaly Score')
    )
    
    # Pie chart of normal vs anomalous users
    if 'prediction' in user_df.columns:
        prediction_counts = user_df['prediction'].value_counts().reset_index()
        prediction_counts.columns = ['Prediction', 'Count']
        
        # Map prediction values to labels
        prediction_counts['Label'] = prediction_counts['Prediction'].map({
            -1: 'Anomalous',
            1: 'Normal'
        })
        
        pie = go.Pie(
            labels=prediction_counts['Label'],
            values=prediction_counts['Count'],
            hole=0.4,
            marker_colors=['#FF6B6B', '#4ECDC4']
        )
        fig.add_trace(pie, row=1, col=1)
        
        # Bar chart of top anomalous users
        if 'user_id' in user_df.columns and 'anomaly_score' in user_df.columns:
            anomalous_users = user_df[user_df['prediction'] == -1]
            if not anomalous_users.empty:
                top_users = anomalous_users.sort_values('anomaly_score', ascending=True).head(5)
                
                bar = go.Bar(
                    x=top_users['user_id'],
                    y=top_users['anomaly_score'],
                    marker_color='#FF6B6B'
                )
                fig.add_trace(bar, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text='User Behavior Analysis',
        showlegend=False,
        height=500
    )
    
    return fig

def create_anomaly_charts(logs_df):
    """
    Create visualizations for anomaly detection in log analysis
    
    Parameters:
    -----------
    logs_df : pd.DataFrame
        DataFrame containing log analysis results
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('Anomaly Categories', 'Confidence Distribution')
    )
    
    # Filter for anomalous logs
    if 'category' in logs_df.columns and 'confidence' in logs_df.columns:
        anomalous_logs = logs_df[logs_df['category'] != 'Normal']
        
        if not anomalous_logs.empty:
            # Pie chart of anomaly categories
            category_counts = anomalous_logs['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            pie = go.Pie(
                labels=category_counts['Category'],
                values=category_counts['Count'],
                hole=0.4,
                marker_colors=px.colors.qualitative.Bold
            )
            fig.add_trace(pie, row=1, col=1)
            
            # Bar chart of confidence distribution
            confidence_bins = [0, 1, 2, 3]
            confidence_labels = ['Low', 'Medium', 'High', 'Critical']
            
            anomalous_logs['confidence_bin'] = pd.cut(
                anomalous_logs['confidence'],
                bins=[0, 1, 2, 3, 4],
                labels=confidence_labels,
                include_lowest=True
            )
            
            confidence_counts = anomalous_logs['confidence_bin'].value_counts().reset_index()
            confidence_counts.columns = ['Confidence', 'Count']
            confidence_counts = confidence_counts.sort_values('Confidence', key=lambda x: pd.Categorical(
                x, categories=confidence_labels, ordered=True
            ))
            
            bar = go.Bar(
                x=confidence_counts['Confidence'],
                y=confidence_counts['Count'],
                marker_color=px.colors.sequential.Plasma_r
            )
            fig.add_trace(bar, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text='Log Anomaly Analysis',
        showlegend=False,
        height=500
    )
    
    return fig