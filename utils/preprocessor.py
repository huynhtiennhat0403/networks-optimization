"""
Input Preprocessor - Validate and preprocess network parameters
Updated for Physical-First Strategy
"""

import json
import os
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputPreprocessor:
    """
    Validates network input parameters.
    Focuses on Physical & Environmental parameters required for the new Model.
    """
    
    def __init__(self, config_path: str = "config/network_ranges.json"):
        self.config_path = config_path
        self.ranges = self._load_ranges()
        
        # --- DANH SÁCH FEATURES CẦN THIẾT (PHYSICAL ONLY) ---
        # Đây là những gì Model THỰC SỰ cần để tính toán
        self.required_features = [
            'User Speed (m/s)',
            'Signal Strength (dBm)',
            'Battery Level (%)',
            'Distance from Base Station (m)',
            'Transmission Power (dBm)',
            'Handover Events',          
            'Power Consumption (mW)',   
            'Network Congestion'
        ]
        
        # Mapping Congestion (Chỉ dùng để validate giá trị hợp lệ)
        self.valid_congestion_values = [
            'Low', 'Medium', 'High', 'low', 'medium', 'high', 
            0, 1, 2, 3 # Chấp nhận cả số
        ]
    
    def _load_ranges(self) -> Dict:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load ranges: {e}")
        return {} 

    def validate_input(self, input_data: Dict) -> Tuple[bool, List[str]]:
        errors = []
        
        # 1. Check Missing Fields (Chỉ check các trường vật lý quan trọng)
        for feature in self.required_features:
            if feature not in input_data:
                errors.append(f"Missing required param: {feature}")

        # 2. Check Values
        if not errors:
            for feature, value in input_data.items():
                # Chỉ validate nếu feature nằm trong tập Physical
                if feature in self.ranges and feature in self.required_features:
                    range_info = self.ranges[feature]
                    
                    # Validate số học
                    if isinstance(value, (int, float)):
                        min_v = range_info.get('min', -float('inf'))
                        max_v = range_info.get('max', float('inf'))
                        # Nới lỏng 20%
                        margin = (max_v - min_v) * 0.2 if max_v != float('inf') else 0
                        
                        if not (min_v - margin <= value <= max_v + margin):
                            logger.warning(f"⚠️ {feature} value {value} is out of typical range")

                    # Validate Network Congestion
                    if feature == 'Network Congestion':
                        if value not in self.valid_congestion_values:
                             errors.append(f"{feature}: Invalid value '{value}'")

        return len(errors) == 0, errors
    
    def preprocess(self, input_data: Dict, validate: bool = True) -> Dict:
        """
        Pass-through clean data.
        Để ModelWrapper lo việc tính toán Feature Engineering phức tạp.
        """
        if validate:
            is_valid, errors = self.validate_input(input_data)
            if not is_valid:
                logger.error(f"Validation failed: {errors}")
                # raise ValueError(f"Validation failed: {errors}")
        
        processed = input_data.copy()
        
        # Đảm bảo số là số
        numeric_cols = [
            'User Speed (m/s)', 'Signal Strength (dBm)', 'Battery Level (%)',
            'Distance from Base Station (m)', 'Transmission Power (dBm)',
            'Handover Events', 'Power Consumption (mW)'
        ]
        
        for col in numeric_cols:
            if col in processed:
                try:
                    processed[col] = float(processed[col])
                except:
                    pass

        return processed

    def create_sample_input(self) -> Dict:
        return {
            'User Speed (m/s)': 15.0,
            'Signal Strength (dBm)': -85.0,
            'Battery Level (%)': 75.0,
            'Distance from Base Station (m)': 500.0,
            'Transmission Power (dBm)': 23.0,
            'Handover Events': 1,
            'Power Consumption (mW)': 400.0,
            'Network Congestion': 'Medium'
        }

if __name__ == "__main__":
    prep = InputPreprocessor()
    sample = prep.create_sample_input()
    print("Valid:", prep.validate_input(sample))