import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class LGBMClassifier:
    """
    Classifier for network intrusion detection
    Trained on the UNSW-NB15 dataset to detect known attack types
    (Using RandomForest as a drop-in replacement for LightGBM due to dependency issues)
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_features = ['proto', 'service', 'state']
        self.attack_categories = {
            0: 'Normal',
            1: 'Generic',
            2: 'Exploits',
            3: 'Fuzzers',
            4: 'DoS',
            5: 'Reconnaissance',
            6: 'Analysis',
            7: 'Backdoor',
            8: 'Shellcode',
            9: 'Worms'
        }
    
    def preprocess_data(self, data):
        """
        Preprocess input data for model prediction
        """
        # Create a copy of the input data
        df = data.copy()
        
        # Handle categorical features
        for feature in self.categorical_features:
            if feature in df.columns:
                # Create label encoder if it doesn't exist
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    # Fit on the current data
                    self.label_encoders[feature].fit(df[feature].astype(str))
                
                # Transform the data
                df[feature] = self.label_encoders[feature].transform(df[feature].astype(str))
        
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df[numeric_cols]
        
        # Replace NaN values with 0
        df_numeric.fillna(0, inplace=True)
        
        # Check if scaler is fitted, if not, fit it
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(self.scaler)
        except:
            self.scaler.fit(df_numeric)
            
        # Scale numeric features
        df_scaled = self.scaler.transform(df_numeric)
        
        return pd.DataFrame(df_scaled, columns=numeric_cols)
    
    def initialize_model(self):
        """
        Initialize and train the model
        """
        # Using RandomForest as a drop-in replacement for LightGBM due to dependency issues
        
        # Create a simple model (in real scenario, this would be trained on UNSW dataset)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Since we don't have the full training data, we'll create a placeholder model
        # In a real implementation, you would load a pre-trained model or train on actual data
        
        # Create dummy data to initialize the model
        X = np.random.rand(100, 10)
        y = np.random.randint(0, len(self.attack_categories), 100)
        
        # Fit the model with dummy data
        self.model.fit(X, y)
        
        return self
    
    def train(self, X, y):
        """
        Train the model with actual data
        """
        # Process features
        X_processed = self.preprocess_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model - RandomForest doesn't use the same params as LGBM
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'model': self.model
        }
    
    def predict(self, X):
        """
        Make predictions with the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        # Process features
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        # Process features
        X_processed = self.preprocess_data(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
            
        import joblib
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from a file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        import joblib
        self.model = joblib.load(filepath)
        
        return self
