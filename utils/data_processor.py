import pandas as pd
import numpy as np
import datetime
import re

class DataProcessor:
    """
    Utility class for processing and preparing data for the models
    """
    
    def __init__(self):
        self.categorical_features = ['proto', 'service', 'state']
    
    def preprocess_network_data(self, data):
        """
        Preprocess network traffic data for anomaly detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw network traffic data
            
        Returns:
        --------
        pd.DataFrame
            Processed data ready for model input
        """
        # Create a copy of the data
        df = data.copy()
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('-')
            else:
                df[col] = df[col].fillna(0)
        
        # Convert categorical features to string
        for feature in self.categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(str)
        
        # Extract datetime features if timestamp is available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Ensure all required columns exist
        required_columns = [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df
    
    def preprocess_user_data(self, data):
        """
        Preprocess user activity data for behavior analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw user activity data
            
        Returns:
        --------
        pd.DataFrame
            Processed user data ready for model input
        """
        # Create a copy of the data
        df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['user_id', 'timestamp', 'action']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'user_id':
                    df[col] = 'unknown_user'
                elif col == 'timestamp':
                    df[col] = datetime.datetime.now()
                else:
                    df[col] = 'unknown'
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract date features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_business_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
        
        return df
    
    def extract_user_behavior_features(self, data):
        """
        Extract features from user activity data for anomaly detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            Processed user activity data
            
        Returns:
        --------
        pd.DataFrame
            Features for the user behavior model
        """
        # Create a copy of the data
        df = data.copy()
        
        # Group by user_id
        user_features = []
        
        for user_id, group in df.groupby('user_id'):
            features = {
                'user_id': user_id,
                'activity_count': len(group),
                'unique_actions': group['action'].nunique(),
                'business_hours_ratio': group['is_business_hours'].mean(),
                'weekend_ratio': group['is_weekend'].mean()
            }
            
            # Add hour distribution features
            for hour in range(24):
                hour_count = len(group[group['hour'] == hour])
                features[f'hour_{hour}_ratio'] = hour_count / len(group) if len(group) > 0 else 0
            
            # Add day of week distribution features
            for day in range(7):
                day_count = len(group[group['day_of_week'] == day])
                features[f'day_{day}_ratio'] = day_count / len(group) if len(group) > 0 else 0
            
            user_features.append(features)
        
        return pd.DataFrame(user_features)
    
    def process_log_messages(self, data):
        """
        Process log messages for unknown threat detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw log data with message column
            
        Returns:
        --------
        pd.DataFrame
            Processed log data
        """
        # Create a copy of the data
        df = data.copy()
        
        # Ensure message column exists
        if 'message' not in df.columns:
            raise ValueError("Log data must contain a 'message' column")
        
        # Fill missing values
        df['message'] = df['message'].fillna('')
        
        # Extract timestamps if available
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}'
        
        def extract_timestamp(message):
            match = re.search(timestamp_pattern, message)
            if match:
                try:
                    return pd.to_datetime(match.group(0))
                except:
                    return pd.NaT
            return pd.NaT
        
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['message'].apply(extract_timestamp)
        
        # Extract IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        
        def extract_ips(message):
            return re.findall(ip_pattern, message)
        
        df['ip_addresses'] = df['message'].apply(extract_ips)
        df['ip_count'] = df['ip_addresses'].apply(len)
        
        # Extract common log levels
        log_levels = ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'CRITICAL', 'FATAL']
        
        def extract_log_level(message):
            for level in log_levels:
                if level in message.upper():
                    return level
            return 'UNKNOWN'
        
        df['log_level'] = df['message'].apply(extract_log_level)
        
        return df
