import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_anomaly_charts(data):
    """
    Create visualizations for anomaly detection results
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing anomaly detection results
        
    Returns:
    --------
    list
        List of plotly figures
    """
    charts = []
    
    # Ensure data is not empty
    if data.empty:
        return charts
    
    # Anomaly distribution
    anomaly_counts = data['prediction'].value_counts().reset_index()
    anomaly_counts.columns = ['Prediction', 'Count']
    anomaly_counts['Prediction'] = anomaly_counts['Prediction'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    
    fig1 = px.pie(
        anomaly_counts, 
        values='Count', 
        names='Prediction',
        title='Anomaly Distribution',
        color='Prediction',
        color_discrete_map={'Normal': '#2ECC71', 'Anomaly': '#E74C3C'},
        hole=0.4
    )
    charts.append(fig1)
    
    # Attack categories if available
    if 'predicted_attack_cat' in data.columns:
        attack_counts = data[data['prediction'] == -1]['predicted_attack_cat'].value_counts().reset_index()
        attack_counts.columns = ['Attack Category', 'Count']
        
        fig2 = px.bar(
            attack_counts,
            x='Attack Category',
            y='Count',
            title='Attack Categories',
            color='Attack Category',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        charts.append(fig2)
    
    # Feature importance if possible
    if 'dur' in data.columns and 'spkts' in data.columns:
        normal = data[data['prediction'] == 1]
        anomaly = data[data['prediction'] == -1]
        
        # Find common numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create subplot grid
        feature_fig = make_subplots(
            rows=len(numeric_cols[:6])//2 + len(numeric_cols[:6])%2, 
            cols=2,
            subplot_titles=[col for col in numeric_cols[:6]]
        )
        
        # Add histograms for each feature
        for i, col in enumerate(numeric_cols[:6]):
            row = i // 2 + 1
            col_idx = i % 2 + 1
            
            feature_fig.add_trace(
                go.Histogram(
                    x=normal[col],
                    name='Normal',
                    opacity=0.7,
                    marker_color='#2ECC71'
                ),
                row=row, col=col_idx
            )
            
            feature_fig.add_trace(
                go.Histogram(
                    x=anomaly[col],
                    name='Anomaly',
                    opacity=0.7,
                    marker_color='#E74C3C'
                ),
                row=row, col=col_idx
            )
        
        feature_fig.update_layout(
            title='Feature Distributions: Normal vs Anomaly',
            barmode='overlay',
            height=800
        )
        
        charts.append(feature_fig)
    
    return charts

def create_user_behavior_charts(data):
    """
    Create visualizations for user behavior analysis results
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing user behavior analysis results
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure for user behavior analysis
    """
    # Create a figure with 1x2 subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['User Behavior Analysis', 'Anomaly Score Distribution']
    )
    
    # User behavior analysis - prediction distribution
    prediction_counts = data['prediction'].value_counts().reset_index()
    prediction_counts.columns = ['Prediction', 'Count']
    prediction_counts['Prediction'] = prediction_counts['Prediction'].apply(lambda x: 'Anomalous' if x == -1 else 'Normal')
    
    fig.add_trace(
        go.Pie(
            labels=prediction_counts['Prediction'],
            values=prediction_counts['Count'],
            hole=0.4,
            marker_colors=['#2ECC71', '#E74C3C']
        ),
        row=1, col=1
    )
    
    # Anomaly score distribution if available
    if 'anomaly_score' in data.columns:
        fig.add_trace(
            go.Histogram(
                x=data['anomaly_score'],
                nbinsx=20,
                marker_color='#3498DB'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

def create_threat_distribution(data):
    """
    Create visualization for threat distribution
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing threat detection results
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure for threat distribution
    """
    # Get attack categories
    if 'attack_cat' in data.columns:
        attack_col = 'attack_cat'
    elif 'predicted_attack_cat' in data.columns:
        attack_col = 'predicted_attack_cat'
    else:
        # Create a default figure
        fig = go.Figure()
        fig.add_annotation(
            text="No attack category data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Count attack categories
    attack_counts = data[attack_col].value_counts().reset_index()
    attack_counts.columns = ['Category', 'Count']
    
    # Create figure
    fig = px.bar(
        attack_counts,
        x='Category',
        y='Count',
        title='Threat Distribution by Category',
        color='Category',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(
        xaxis_title='Attack Category',
        yaxis_title='Count',
        height=500
    )
    
    return fig
