"""
Model Wrapper - API wrapper for ML model
Handles model loading, preprocessing, and prediction
Updated for: Physical-First Strategy (No Throughput/Latency)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
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
                 base_dir: str = None,
                 model_name: str = "rf_final_model.pkl",
                 scaler_name: str = "minmax_scaler.pkl",
                 feature_info_name: str = "feature_info.pkl"):
        
        # --- 1. XÂY DỰNG ĐƯỜNG DẪN ---
        if base_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        logger.info(f"ModelWrapper initializing with base_dir: {base_dir}")
        
        models_dir = os.path.join(base_dir, "models")
        
        self.model_path = os.path.join(models_dir, model_name)
        self.scaler_path = os.path.join(models_dir, scaler_name)
        self.feature_info_path = os.path.join(models_dir, feature_info_name)
        
        self.model = None
        self.scaler = None
        self.feature_info = None
        self.feature_names = []
        
        # Mapping Label Output
        self.label_map = {0: 'Poor', 1: 'Moderate', 2: 'Good'}
        
        self._load_model()
    
    def _load_model(self):
        """Load model, scaler, and feature info from disk"""
        try:
            logger.info("Loading model and preprocessors...")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            # Load Model
            self.model = joblib.load(self.model_path)
            
            # Load Scaler
            self.scaler = joblib.load(self.scaler_path)
            
            # Load Feature Info
            if os.path.exists(self.feature_info_path):
                self.feature_info = joblib.load(self.feature_info_path)
                all_cols = self.feature_info.get('all_features', [])
                target_col = 'RF Link Quality'
                self.feature_names = [col for col in all_cols if col != target_col]
            else:
                logger.warning("Feature info not found! Using fallback list.")
                # Fallback list MỚI (Khớp với processing_data.py)
                self.feature_names = [
                    'User Speed (m/s)', 
                    'Signal Strength (dBm)', 
                    'Battery Level (%)',
                    'Distance from Base Station (m)', 
                    'Handover Events',
                    'Power Consumption (mW)',
                    'Transmission Power (dBm)',
                    'Network Congestion Score', # Feature đã xử lý
                    'Mobility_Impact',          # Feature mới
                    'Signal_Quality_Index',     # Feature mới
                    'Device_Stress_Level',      # Feature mới
                    'Log_Distance'              # Feature mới
                ]

            logger.info(f"✅ Components loaded. Expecting {len(self.feature_names)} features.")
            logger.info(f"📋 Feature list: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise

    def preprocess_input(self, raw_input: Dict) -> np.ndarray:
        """
        Feature Engineering & Scaling
        Logic phải KHỚP 100% với processing_data.py
        """
        # 1. Chuyển dict thành DataFrame
        df = pd.DataFrame([raw_input])
        
        # 2. Xử lý Network Congestion -> Network Congestion Score (1, 2, 3)
        # Input có thể là: 'Low'/'Medium'/'High' (từ Client) HOẶC 0/1/2 (từ SmartEstimator cũ)
        
        def map_congestion(val):
            # Nếu là chuỗi
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower == 'low': return 1
                if val_lower == 'medium': return 2
                if val_lower == 'high': return 3
            # Nếu là số (0,1,2 -> 1,2,3)
            if isinstance(val, (int, float)):
                if val == 0: return 1
                if val == 1: return 2
                if val == 2: return 3
                if val in [1, 2, 3]: return int(val) # Đã chuẩn
            return 2 # Default Medium

        if 'Network Congestion' in df.columns:
            df['Network Congestion Score'] = df['Network Congestion'].apply(map_congestion)
        else:
            df['Network Congestion Score'] = 2 # Default

        # 3. --- TÍNH TOÁN FEATURE MỚI (Feature Engineering) ---
        # Đảm bảo handle trường hợp thiếu key bằng .get() với giá trị default an toàn
        
        # F1: Mobility Impact
        df['Mobility_Impact'] = df.get('User Speed (m/s)', 0) * (df.get('Handover Events', 0) + 1)
        
        # F2: Signal Quality Index
        df['Signal_Quality_Index'] = df.get('Signal Strength (dBm)', -85) * df['Network Congestion Score']
        
        # F3: Device Stress Level
        df['Device_Stress_Level'] = df.get('Power Consumption (mW)', 500) / (df.get('Battery Level (%)', 50) + 1)
        
        # F4: Log Distance
        df['Log_Distance'] = np.log1p(df.get('Distance from Base Station (m)', 100))
        
        # 4. Sắp xếp cột đúng thứ tự train
        final_df = pd.DataFrame()
        for col in self.feature_names:
            if col in df.columns:
                final_df[col] = df[col]
            else:
                logger.warning(f"⚠️ Missing feature '{col}', filling with 0")
                final_df[col] = 0.0

        # 5. Scale dữ liệu
        return self.scaler.transform(final_df)

    def predict(self, raw_input: Dict) -> Dict:
        try:
            X = self.preprocess_input(raw_input)
            
            y_pred = self.model.predict(X)[0]
            y_proba = self.model.predict_proba(X)[0]
            
            result = {
                'prediction': int(y_pred),
                'prediction_label': self.label_map.get(y_pred, "Unknown"),
                'confidence': float(y_proba[y_pred]),
                'probabilities': {
                    'Poor': float(y_proba[0]),
                    'Moderate': float(y_proba[1]),
                    'Good': float(y_proba[2])
                }
            }
            return result
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return {
            self.feature_names[i]: float(importances[i])
            for i in indices if i < len(self.feature_names)
        }

if __name__ == "__main__":
    wrapper = ModelWrapper()
    sample = {
        'User Speed (m/s)': 15.0,
        'Signal Strength (dBm)': -40.0,
        'Battery Level (%)': 100.0,
        'Distance from Base Station (m)': 500.0,
        'Transmission Power (dBm)': 23.0,
        'Handover Events': 1,
        'Power Consumption (mW)': 400.0,
        'Network Congestion': 'Low' # Test chữ
    }
    print(wrapper.predict(sample))