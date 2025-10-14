"""
Model Wrapper - API wrapper for ML model
Handles model loading, preprocessing, and prediction
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWrapper:
    """
    Wrapper class for the trained Random Forest model
    Handles loading, preprocessing, and prediction
    """
    
    def __init__(self, 
                 model_path: str = "models/rf_model.pkl",
                 scaler_path: str = "models/minmax_scaler.pkl",
                 encoder_path: str = "models/onehot_encoder.pkl"):
        """
        Initialize the model wrapper
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to MinMaxScaler
            encoder_path: Path to OneHotEncoder
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        
        # Label mapping
        self.label_map = {0: 'Poor', 1: 'Moderate', 2: 'Good'}
        
        # Load model and preprocessors
        self._load_model()
    
    
    def _load_model(self):
        """Load model, scaler, and encoder from disk"""
        try:
            logger.info("Loading model and preprocessors...")
            
            # Check if files exist
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            if not os.path.exists(self.encoder_path):
                raise FileNotFoundError(f"Encoder file not found: {self.encoder_path}")
            
            # Load model
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Load scaler
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"✅ Scaler loaded: {type(self.scaler).__name__}")
            
            # Load encoder
            self.encoder = joblib.load(self.encoder_path)
            logger.info(f"✅ Encoder loaded: {type(self.encoder).__name__}")
            
            # Get feature names
            self._extract_feature_names()
            
            logger.info("🎉 All components loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    
    def _extract_feature_names(self):
        """Extract feature names from training data"""
        # This should match the feature order from training
        # You can also save this during training for better consistency
        
        # Continuous features (from your dataset)
        continuous_features = [
            'User Speed (m/s)',
            'User Direction (degrees)',
            'Handover Events',
            'Distance from Base Station (m)',
            'Signal Strength (dBm)',
            'SNR (dB)',
            'BER',
            'PDR (%)',
            'Throughput (Mbps)',
            'Latency (ms)',
            'Retransmission Count',
            'Power Consumption (mW)',
            'Battery Level (%)',
            'Transmission Power (dBm)'
        ]
        
        # Categorical features
        categorical_features = ['Network Congestion']
        
        # One-hot encoded features (Modulation Scheme)
        modulation_schemes = self.encoder.get_feature_names_out(['Modulation Scheme'])
        
        # Combine all features in correct order
        self.feature_names = continuous_features + categorical_features + list(modulation_schemes)
        
        logger.info(f"📊 Total features: {len(self.feature_names)}")
    
    
    def preprocess_input(self, raw_input: Dict) -> np.ndarray:
        """
        Preprocess raw input data
        
        Args:
            raw_input: Dictionary with network parameters
            
        Returns:
            Preprocessed numpy array ready for prediction
        """
        try:
            # Create DataFrame with correct feature order
            df = pd.DataFrame([raw_input])
            
            # Handle Modulation Scheme encoding
            if 'Modulation Scheme' in df.columns:
                # One-hot encode
                mod_encoded = self.encoder.transform(df[['Modulation Scheme']])
                mod_df = pd.DataFrame(
                    mod_encoded,
                    columns=self.encoder.get_feature_names_out(['Modulation Scheme'])
                )
                
                # Drop original column and add encoded
                df = df.drop(columns=['Modulation Scheme'])
                df = pd.concat([df, mod_df], axis=1)
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Default value for missing features
            
            # Reorder columns to match training
            df = df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {str(e)}")
            raise
    
    
    def predict(self, raw_input: Dict) -> Dict:
        """
        Make prediction from raw input
        
        Args:
            raw_input: Dictionary with network parameters
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        try:
            # Preprocess input
            X = self.preprocess_input(raw_input)
            
            # Predict class
            y_pred = self.model.predict(X)[0]
            
            # Get probabilities
            y_proba = self.model.predict_proba(X)[0]
            
            # Format result
            result = {
                'prediction': int(y_pred),
                'prediction_label': self.label_map[y_pred],
                'confidence': float(y_proba[y_pred]),
                'probabilities': {
                    'Poor': float(y_proba[0]),
                    'Moderate': float(y_proba[1]),
                    'Good': float(y_proba[2])
                },
                'all_probabilities': y_proba.tolist()
            }
            
            logger.info(f"✅ Prediction: {result['prediction_label']} ({result['confidence']:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise
    
    
    def predict_batch(self, raw_inputs: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple inputs
        
        Args:
            raw_inputs: List of input dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for input_data in raw_inputs:
            result = self.predict(input_data)
            results.append(result)
        
        return results
    
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model doesn't have feature_importances_ attribute")
            return {}
        
        importances = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        feature_importance = {
            self.feature_names[i]: float(importances[i])
            for i in indices
        }
        
        return feature_importance
    
    
    def health_check(self) -> Dict:
        """
        Check if model is loaded and ready
        
        Returns:
            Dictionary with health status
        """
        status = {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'encoder_loaded': self.encoder is not None,
            'features_count': len(self.feature_names) if self.feature_names else 0,
            'model_type': type(self.model).__name__ if self.model else None
        }
        
        status['ready'] = all([
            status['model_loaded'],
            status['scaler_loaded'],
            status['encoder_loaded']
        ])
        
        return status


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == "__main__":
    # Initialize wrapper
    wrapper = ModelWrapper()
    
    # Health check
    health = wrapper.health_check()
    print("\n🏥 Health Check:")
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    # Example input
    sample_input = {
        'User Speed (m/s)': 15.5,
        'User Direction (degrees)': 180.0,
        'Handover Events': 2,
        'Distance from Base Station (m)': 150.0,
        'Signal Strength (dBm)': -75.0,
        'SNR (dB)': 20.0,
        'BER': 0.001,
        'Modulation Scheme': 'QPSK',
        'PDR (%)': 95.0,
        'Network Congestion': 1,  # Low=0, Medium=1, High=2
        'Throughput (Mbps)': 50.0,
        'Latency (ms)': 30.0,
        'Retransmission Count': 3,
        'Power Consumption (mW)': 500.0,
        'Battery Level (%)': 80.0,
        'Transmission Power (dBm)': 20.0
    }
    
    # Make prediction
    print("\n🔮 Making prediction...")
    result = wrapper.predict(sample_input)
    
    print("\n📊 Prediction Result:")
    print(f"  Prediction: {result['prediction_label']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"    {label}: {prob:.2%}")
    
    # Feature importance
    print("\n🔍 Top 10 Important Features:")
    importance = wrapper.get_feature_importance(top_n=10)
    for i, (feature, score) in enumerate(importance.items(), 1):
        print(f"  {i:2d}. {feature:<35} {score:.4f}")