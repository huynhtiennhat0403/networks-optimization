"""
Input Preprocessor - Validate and preprocess network parameters
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputPreprocessor:
    """
    Validates and preprocesses network input parameters
    """
    
    def __init__(self, config_path: str = "config/network_ranges.json"):
        """
        Initialize preprocessor with validation ranges
        
        Args:
            config_path: Path to network ranges configuration
        """
        self.config_path = config_path
        self.ranges = self._load_ranges()
        
        # Define expected features
        self.continuous_features = [
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
        
        self.categorical_features = ['Modulation Scheme', 'Network Congestion']
        
        # Network Congestion mapping
        self.congestion_map = {
            'Low': 0,
            'Medium': 1,
            'High': 2,
            0: 0,
            1: 1,
            2: 2
        }
    
    
    def _load_ranges(self) -> Dict:
        """Load validation ranges from config file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}")
                logger.warning("Using default ranges...")
                return self._get_default_ranges()
            
            with open(self.config_path, 'r') as f:
                ranges = json.load(f)
            
            logger.info(f"✅ Loaded ranges from {self.config_path}")
            return ranges
            
        except Exception as e:
            logger.error(f"❌ Error loading ranges: {str(e)}")
            logger.warning("Using default ranges...")
            return self._get_default_ranges()
    
    
    def _get_default_ranges(self) -> Dict:
        """Default ranges based on your data"""
        return {
            "User Speed (m/s)": {"min": 0.0, "max": 30.0},
            "User Direction (degrees)": {"min": 0.0, "max": 360.0},
            "Handover Events": {"min": 0, "max": 4},
            "Distance from Base Station (m)": {"min": 10.0, "max": 1000.0},
            "Signal Strength (dBm)": {"min": -100.0, "max": -40.0},
            "SNR (dB)": {"min": 5.0, "max": 30.0},
            "BER": {"min": 0.0001, "max": 0.05},
            "PDR (%)": {"min": 50.0, "max": 100.0},
            "Throughput (Mbps)": {"min": 1.0, "max": 100.0},
            "Latency (ms)": {"min": 1.0, "max": 100.0},
            "Retransmission Count": {"min": 0, "max": 9},
            "Power Consumption (mW)": {"min": 100.0, "max": 1000.0},
            "Battery Level (%)": {"min": 5.0, "max": 100.0},
            "Transmission Power (dBm)": {"min": 0.0, "max": 30.0},
            "Modulation Scheme": {"type": "categorical", "values": ["BPSK", "QPSK", "16-QAM", "64-QAM"]},
            "Network Congestion": {"type": "categorical", "values": ["Low", "Medium", "High"]}
        }
    
    
    def validate_input(self, input_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate input data against defined ranges
        
        Args:
            input_data: Dictionary with network parameters
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check continuous features
        for feature in self.continuous_features:
            if feature not in input_data:
                errors.append(f"Missing required field: {feature}")
                continue
            
            value = input_data[feature]
            
            # Check type
            if not isinstance(value, (int, float)):
                errors.append(f"{feature}: Expected number, got {type(value).__name__}")
                continue
            
            # Check range
            if feature in self.ranges:
                min_val = self.ranges[feature].get('min', float('-inf'))
                max_val = self.ranges[feature].get('max', float('inf'))
                
                if value < min_val or value > max_val:
                    errors.append(
                        f"{feature}: Value {value} out of range [{min_val}, {max_val}]"
                    )
        
        # Check categorical features
        for feature in self.categorical_features:
            if feature not in input_data:
                errors.append(f"Missing required field: {feature}")
                continue
            
            value = input_data[feature]
            
            if feature in self.ranges:
                valid_values = self.ranges[feature].get('values', [])
                
                if feature == 'Network Congestion':
                    # Accept both string and numeric values
                    if value not in valid_values and value not in [0, 1, 2]:
                        errors.append(
                            f"{feature}: Invalid value '{value}'. Must be one of {valid_values} or 0/1/2"
                        )
                else:
                    if value not in valid_values:
                        errors.append(
                            f"{feature}: Invalid value '{value}'. Must be one of {valid_values}"
                        )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"❌ Validation failed with {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("✅ Validation passed")
        
        return is_valid, errors
    
    
    def preprocess(self, input_data: Dict, validate: bool = True) -> Dict:
        """
        Preprocess and clean input data
        
        Args:
            input_data: Raw input dictionary
            validate: Whether to validate before preprocessing
            
        Returns:
            Cleaned and preprocessed dictionary
        """
        # Validate first
        if validate:
            is_valid, errors = self.validate_input(input_data)
            if not is_valid:
                raise ValueError(f"Input validation failed: {errors}")
        
        # Create copy to avoid modifying original
        processed = input_data.copy()
        
        # Convert Network Congestion to numeric if needed
        if 'Network Congestion' in processed:
            congestion_value = processed['Network Congestion']
            if congestion_value in self.congestion_map:
                processed['Network Congestion'] = self.congestion_map[congestion_value]
            else:
                # If not in map, try to keep as is (might be already numeric)
                try:
                    processed['Network Congestion'] = int(congestion_value)
                except:
                    raise ValueError(f"Invalid Network Congestion value: {congestion_value}")
        
        # Round integer features
        integer_features = ['Handover Events', 'Retransmission Count']
        for feature in integer_features:
            if feature in processed:
                processed[feature] = int(round(processed[feature]))
        
        # Clip continuous values to valid ranges
        for feature in self.continuous_features:
            if feature in processed and feature in self.ranges:
                min_val = self.ranges[feature].get('min')
                max_val = self.ranges[feature].get('max')
                
                if min_val is not None and max_val is not None:
                    original_value = processed[feature]
                    processed[feature] = max(min_val, min(max_val, processed[feature]))
                    
                    if processed[feature] != original_value:
                        logger.warning(
                            f"⚠️  {feature}: Clipped {original_value} to {processed[feature]}"
                        )
        
        return processed
    
    
    def get_feature_info(self, feature_name: str) -> Dict:
        """
        Get information about a specific feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with feature information
        """
        if feature_name not in self.ranges:
            return {'error': f'Feature {feature_name} not found'}
        
        info = self.ranges[feature_name].copy()
        info['name'] = feature_name
        
        # Add feature type
        if feature_name in self.continuous_features:
            info['type'] = 'continuous'
        elif feature_name in self.categorical_features:
            info['type'] = 'categorical'
        
        return info
    
    
    def get_all_features_info(self) -> Dict[str, Dict]:
        """
        Get information about all features
        
        Returns:
            Dictionary mapping feature names to their info
        """
        all_info = {}
        
        for feature in self.continuous_features + self.categorical_features:
            all_info[feature] = self.get_feature_info(feature)
        
        return all_info
    
    
    def suggest_value(self, feature_name: str, quality: str = 'moderate') -> Any:
        """
        Suggest a reasonable value for a feature based on desired quality
        
        Args:
            feature_name: Name of the feature
            quality: Desired quality level ('poor', 'moderate', 'good')
            
        Returns:
            Suggested value
        """
        if feature_name not in self.ranges:
            return None
        
        feature_info = self.ranges[feature_name]
        
        # Categorical features
        if feature_info.get('type') == 'categorical':
            if feature_name == 'Modulation Scheme':
                mapping = {
                    'poor': 'BPSK',
                    'moderate': 'QPSK',
                    'good': '64-QAM'
                }
                return mapping.get(quality.lower(), 'QPSK')
            
            elif feature_name == 'Network Congestion':
                mapping = {
                    'poor': 'High',
                    'moderate': 'Medium',
                    'good': 'Low'
                }
                return mapping.get(quality.lower(), 'Medium')
        
        # Continuous features
        else:
            min_val = feature_info.get('min', 0)
            max_val = feature_info.get('max', 100)
            mean_val = feature_info.get('mean', (min_val + max_val) / 2)
            
            # Features where HIGHER is BETTER
            higher_better = [
                'SNR (dB)',
                'PDR (%)',
                'Throughput (Mbps)',
                'Battery Level (%)',
                'Transmission Power (dBm)'
            ]
            
            # Features where LOWER is BETTER
            lower_better = [
                'BER',
                'Latency (ms)',
                'Distance from Base Station (m)',
                'Retransmission Count',
                'Power Consumption (mW)'
            ]
            
            # Features where it depends (use mean)
            neutral = [
                'User Speed (m/s)',
                'User Direction (degrees)',
                'Handover Events',
                'Signal Strength (dBm)'  # More negative = worse, but can't be too high
            ]
            
            if feature_name in higher_better:
                mapping = {
                    'poor': min_val + (max_val - min_val) * 0.2,
                    'moderate': mean_val,
                    'good': min_val + (max_val - min_val) * 0.8
                }
            elif feature_name in lower_better:
                mapping = {
                    'poor': min_val + (max_val - min_val) * 0.8,
                    'moderate': mean_val,
                    'good': min_val + (max_val - min_val) * 0.2
                }
            else:
                # Use mean for neutral features
                return mean_val
            
            value = mapping.get(quality.lower(), mean_val)
            
            # Round if integer feature
            if feature_name in ['Handover Events', 'Retransmission Count']:
                value = int(round(value))
            
            return value
    
    
    def create_sample_input(self, quality: str = 'moderate') -> Dict:
        """
        Create a complete sample input with reasonable values
        
        Args:
            quality: Desired quality level ('poor', 'moderate', 'good')
            
        Returns:
            Dictionary with all required features
        """
        sample = {}
        
        for feature in self.continuous_features + self.categorical_features:
            sample[feature] = self.suggest_value(feature, quality)
        
        return sample


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = InputPreprocessor()
    
    print("="*80)
    print("PREPROCESSOR TEST")
    print("="*80)
    
    # Test 1: Valid input
    print("\n🧪 Test 1: Valid Input")
    valid_input = {
        'User Speed (m/s)': 15.5,
        'User Direction (degrees)': 180.0,
        'Handover Events': 2,
        'Distance from Base Station (m)': 150.0,
        'Signal Strength (dBm)': -75.0,
        'SNR (dB)': 20.0,
        'BER': 0.001,
        'Modulation Scheme': 'QPSK',
        'PDR (%)': 95.0,
        'Network Congestion': 'Medium',
        'Throughput (Mbps)': 50.0,
        'Latency (ms)': 30.0,
        'Retransmission Count': 3,
        'Power Consumption (mW)': 500.0,
        'Battery Level (%)': 80.0,
        'Transmission Power (dBm)': 20.0
    }
    
    is_valid, errors = preprocessor.validate_input(valid_input)
    if is_valid:
        processed = preprocessor.preprocess(valid_input)
        print("✅ Validation passed!")
        print(f"Network Congestion converted: {processed['Network Congestion']}")
    else:
        print(f"❌ Validation failed: {errors}")
    
    # Test 2: Invalid input (out of range)
    print("\n🧪 Test 2: Out of Range Values")
    invalid_input = valid_input.copy()
    invalid_input['SNR (dB)'] = 100.0  # Too high
    invalid_input['BER'] = 1.5  # Too high
    
    is_valid, errors = preprocessor.validate_input(invalid_input)
    print(f"Valid: {is_valid}")
    if not is_valid:
        print("Errors found:")
        for error in errors:
            print(f"  - {error}")
    
    # Test 3: Missing fields
    print("\n🧪 Test 3: Missing Fields")
    incomplete_input = {'SNR (dB)': 20.0}
    is_valid, errors = preprocessor.validate_input(incomplete_input)
    print(f"Valid: {is_valid}")
    print(f"Missing {len(errors)} fields")
    
    # Test 4: Generate sample inputs
    print("\n🧪 Test 4: Generate Sample Inputs")
    
    for quality in ['poor', 'moderate', 'good']:
        print(f"\n{quality.upper()} Quality Sample:")
        sample = preprocessor.create_sample_input(quality)
        print(f"  SNR: {sample['SNR (dB)']:.2f} dB")
        print(f"  Throughput: {sample['Throughput (Mbps)']:.2f} Mbps")
        print(f"  Latency: {sample['Latency (ms)']:.2f} ms")
        print(f"  Modulation: {sample['Modulation Scheme']}")
        print(f"  Congestion: {sample['Network Congestion']}")
    
    # Test 5: Feature info
    print("\n🧪 Test 5: Feature Information")
    snr_info = preprocessor.get_feature_info('SNR (dB)')
    print(f"SNR (dB) Info:")
    print(f"  Type: {snr_info.get('type')}")
    print(f"  Range: [{snr_info.get('min'):.2f}, {snr_info.get('max'):.2f}]")
    print(f"  Mean: {snr_info.get('mean', 'N/A')}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)