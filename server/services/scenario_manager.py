"""
Scenario Manager - Manages pre-defined network scenarios for Vietnam context
Mode 2: Scenario Simulation
"""

import random
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioManager:
    """
    Manages realistic network scenarios based on Vietnam urban contexts
    Each scenario has pre-defined parameter ranges matching the dataset
    """
    
    def __init__(self):
        """Initialize with 5 realistic Vietnam scenarios"""
        self.scenarios = self._create_scenarios()
        logger.info(f"✅ Initialized {len(self.scenarios)} scenarios")
    
    
    def _create_scenarios(self) -> Dict[int, Dict]:
        """
        Create 5 realistic scenarios for Vietnam
        Each scenario maps to typical network conditions
        """
        
        scenarios = {
            1: {
                "id": 1,
                "name": "🚌 Di chuyển xe bus/Grab (TP.HCM)",
                "name_en": "Urban Commute (Bus/Grab)",
                "description": "Gọi video call khi di chuyển trong thành phố. Tốc độ 20-40 km/h, nhiều handover.",
                "icon": "🚌",
                "typical_use": "Video call, browsing, navigation",
                "location": "Quận 1, Quận 3, Bình Thạnh",
                "expected_quality": "Moderate",
                "parameters": {
                    'User Speed (m/s)': 8.5,  # ~30 km/h
                    'User Direction (degrees)': 90.0,
                    'Handover Events': 3,
                    'Distance from Base Station (m)': 250.0,
                    'Signal Strength (dBm)': -78.0,
                    'SNR (dB)': 15.0,
                    'BER': 0.008,
                    'Modulation Scheme': 'QPSK',
                    'PDR (%)': 88.0,
                    'Network Congestion': 1,  # Medium
                    'Throughput (Mbps)': 35.0,
                    'Latency (ms)': 45.0,
                    'Retransmission Count': 4,
                    'Power Consumption (mW)': 550.0,
                    'Battery Level (%)': 65.0,
                    'Transmission Power (dBm)': 23.0
                },
                "challenges": [
                    "Nhiều tòa nhà cao tầng che chắn",
                    "Handover liên tục giữa base stations",
                    "Traffic congestion ảnh hưởng tốc độ di chuyển"
                ]
            },
            
            2: {
                "id": 2,
                "name": "🏢 Văn phòng tòa nhà cao tầng",
                "name_en": "High-rise Office Building",
                "description": "Họp online tại Bitexco, Landmark 81. Tốc độ 0 m/s, nhiều thiết bị cùng kết nối.",
                "icon": "🏢",
                "typical_use": "Video conference, cloud upload, file sharing",
                "location": "Bitexco, Landmark 81, Vietcombank Tower",
                "expected_quality": "Good",
                "parameters": {
                    'User Speed (m/s)': 0.5,  # Gần như đứng yên
                    'User Direction (degrees)': 0.0,
                    'Handover Events': 0,
                    'Distance from Base Station (m)': 120.0,
                    'Signal Strength (dBm)': -68.0,
                    'SNR (dB)': 22.0,
                    'BER': 0.002,
                    'Modulation Scheme': '64-QAM',
                    'PDR (%)': 96.0,
                    'Network Congestion': 1,  # Medium (nhiều người dùng)
                    'Throughput (Mbps)': 65.0,
                    'Latency (ms)': 25.0,
                    'Retransmission Count': 2,
                    'Power Consumption (mW)': 400.0,
                    'Battery Level (%)': 85.0,  # Thường có sạc
                    'Transmission Power (dBm)': 18.0
                },
                "challenges": [
                    "Tường bê tông dày",
                    "Nhiều thiết bị cùng kết nối",
                    "Tầng cao có thể ảnh hưởng tín hiệu"
                ]
            },
            
            3: {
                "id": 3,
                "name": "☕ Quán cafe đông khách",
                "name_en": "Crowded Cafe",
                "description": "Remote work tại The Coffee House, Highlands. Nhiều người dùng chung mạng.",
                "icon": "☕",
                "typical_use": "Video call, upload files, web browsing",
                "location": "The Coffee House, Highlands Coffee, Starbucks",
                "expected_quality": "Moderate",
                "parameters": {
                    'User Speed (m/s)': 0.2,  # Ngồi yên
                    'User Direction (degrees)': 0.0,
                    'Handover Events': 0,
                    'Distance from Base Station (m)': 80.0,
                    'Signal Strength (dBm)': -72.0,
                    'SNR (dB)': 16.0,  # Nhiễu cao do đông người
                    'BER': 0.006,
                    'Modulation Scheme': 'QPSK',
                    'PDR (%)': 90.0,
                    'Network Congestion': 2,  # High
                    'Throughput (Mbps)': 28.0,
                    'Latency (ms)': 55.0,
                    'Retransmission Count': 5,
                    'Power Consumption (mW)': 480.0,
                    'Battery Level (%)': 55.0,
                    'Transmission Power (dBm)': 20.0
                },
                "challenges": [
                    "Quá nhiều thiết bị cùng kết nối WiFi/4G",
                    "Nhiễu cao do mật độ người dùng",
                    "Giờ cao điểm (14h-17h) quá tải"
                ]
            },
            
            4: {
                "id": 4,
                "name": "🛣️ Cao tốc TP.HCM - Long Thành",
                "name_en": "Highway (HCM - Long Thanh)",
                "description": "Navigation/livestream trên cao tốc. Tốc độ 80-100 km/h, handover nhanh.",
                "icon": "🛣️",
                "typical_use": "Navigation, music streaming, messaging",
                "location": "Cao tốc TP.HCM - Long Thành, TP.HCM - Trung Lương",
                "expected_quality": "Poor",
                "parameters": {
                    'User Speed (m/s)': 25.0,  # ~90 km/h
                    'User Direction (degrees)': 45.0,
                    'Handover Events': 4,
                    'Distance from Base Station (m)': 650.0,  # Xa base station
                    'Signal Strength (dBm)': -92.0,
                    'SNR (dB)': 10.0,
                    'BER': 0.022,
                    'Modulation Scheme': 'BPSK',
                    'PDR (%)': 75.0,
                    'Network Congestion': 0,  # Low (ít người dùng)
                    'Throughput (Mbps)': 18.0,
                    'Latency (ms)': 78.0,
                    'Retransmission Count': 7,
                    'Power Consumption (mW)': 780.0,  # Cao do tìm sóng
                    'Battery Level (%)': 40.0,
                    'Transmission Power (dBm)': 28.0
                },
                "challenges": [
                    "Khoảng cách xa base station",
                    "Handover rất nhanh",
                    "Đôi khi mất sóng hoàn toàn",
                    "Battery drain nhanh"
                ]
            },
            
            5: {
                "id": 5,
                "name": "🎉 Sự kiện đông người (Concert/Lễ hội)",
                "name_en": "Crowded Event (Concert/Festival)",
                "description": "Concert, lễ hội tại Phố đi bộ, Landmark. Hàng nghìn người cùng lúc.",
                "icon": "🎉",
                "typical_use": "Messaging, social media, taking photos",
                "location": "Phố đi bộ Nguyễn Huệ, Landmark 81, Đầm Sen",
                "expected_quality": "Poor",
                "parameters": {
                    'User Speed (m/s)': 0.8,  # Di chuyển chậm trong đám đông
                    'User Direction (degrees)': 180.0,
                    'Handover Events': 1,
                    'Distance from Base Station (m)': 100.0,
                    'Signal Strength (dBm)': -85.0,
                    'SNR (dB)': 8.0,  # Rất nhiễu
                    'BER': 0.035,
                    'Modulation Scheme': 'BPSK',
                    'PDR (%)': 62.0,  # Packet loss cao
                    'Network Congestion': 2,  # High
                    'Throughput (Mbps)': 8.0,  # Rất thấp
                    'Latency (ms)': 95.0,  # Rất cao
                    'Retransmission Count': 9,
                    'Power Consumption (mW)': 850.0,
                    'Battery Level (%)': 30.0,
                    'Transmission Power (dBm)': 29.0
                },
                "challenges": [
                    "Quá tải mạng cực kỳ nghiêm trọng",
                    "Hàng nghìn người cùng kết nối",
                    "Gửi tin nhắn/ảnh mất rất lâu",
                    "Có thể không gọi điện được"
                ]
            }
        }
        
        return scenarios
    
    
    def get_scenario(self, scenario_id: int) -> Optional[Dict]:
        """
        Get a specific scenario by ID
        
        Args:
            scenario_id: Scenario ID (1-5)
            
        Returns:
            Scenario dictionary or None if not found
        """
        return self.scenarios.get(scenario_id)
    
    
    def get_all_scenarios(self) -> List[Dict]:
        """
        Get all scenarios as a list
        
        Returns:
            List of all scenarios
        """
        return list(self.scenarios.values())
    
    
    def get_scenario_summary(self) -> List[Dict]:
        """
        Get simplified scenario list (for UI display)
        
        Returns:
            List of scenario summaries
        """
        summaries = []
        for scenario in self.scenarios.values():
            summaries.append({
                'id': scenario['id'],
                'name': scenario['name'],
                'icon': scenario['icon'],
                'description': scenario['description'],
                'expected_quality': scenario['expected_quality']
            })
        return summaries
    
    
    def add_randomness(self, scenario_id: int, variation: float = 0.1) -> Dict:
        """
        Add random variation to scenario parameters (for realistic simulation)
        
        Args:
            scenario_id: Scenario ID
            variation: Variation percentage (0.1 = ±10%)
            
        Returns:
            Scenario with randomized parameters
        """
        scenario = self.get_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        # Create copy
        randomized = scenario.copy()
        randomized['parameters'] = scenario['parameters'].copy()
        
        # Add variation to continuous parameters
        continuous_params = [
            'User Speed (m/s)',
            'Distance from Base Station (m)',
            'Signal Strength (dBm)',
            'SNR (dB)',
            'BER',
            'PDR (%)',
            'Throughput (Mbps)',
            'Latency (ms)',
            'Power Consumption (mW)',
            'Battery Level (%)',
            'Transmission Power (dBm)'
        ]
        
        for param in continuous_params:
            if param in randomized['parameters']:
                original_value = randomized['parameters'][param]
                
                # Add random variation
                random_factor = random.uniform(1 - variation, 1 + variation)
                new_value = original_value * random_factor
                
                # Ensure within reasonable bounds
                if param == 'User Speed (m/s)':
                    new_value = max(0, min(30, new_value))
                elif param == 'Signal Strength (dBm)':
                    new_value = max(-100, min(-40, new_value))
                elif param == 'SNR (dB)':
                    new_value = max(5, min(30, new_value))
                elif param == 'BER':
                    new_value = max(0.0001, min(0.05, new_value))
                elif param == 'PDR (%)':
                    new_value = max(50, min(100, new_value))
                elif param == 'Throughput (Mbps)':
                    new_value = max(1, min(100, new_value))
                elif param == 'Latency (ms)':
                    new_value = max(1, min(100, new_value))
                elif param == 'Battery Level (%)':
                    new_value = max(5, min(100, new_value))
                
                randomized['parameters'][param] = round(new_value, 2)
        
        # Sometimes vary handover events
        if random.random() < 0.3:  # 30% chance
            randomized['parameters']['Handover Events'] = random.randint(0, 4)
        
        return randomized


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == "__main__":
    print("="*80)
    print("SCENARIO MANAGER TEST")
    print("="*80)
    
    manager = ScenarioManager()
    
    # Test 1: List all scenarios
    print("\n📋 All Scenarios:")
    for scenario in manager.get_scenario_summary():
        print(f"\n{scenario['icon']} Scenario {scenario['id']}: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Expected Quality: {scenario['expected_quality']}")
    
    # Test 2: Get specific scenario
    print("\n" + "="*80)
    print("🔍 Detailed Scenario 1 (Urban Commute):")
    scenario1 = manager.get_scenario(1)
    print(f"\nName: {scenario1['name']}")
    print(f"Location: {scenario1['location']}")
    print(f"\nNetwork Parameters:")
    for key, value in scenario1['parameters'].items():
        print(f"  {key}: {value}")
    print(f"\nChallenges:")
    for challenge in scenario1['challenges']:
        print(f"  - {challenge}")
    
    # Test 3: Randomized scenario
    print("\n" + "="*80)
    print("🎲 Randomized Scenario 3 (Cafe):")
    randomized = manager.add_randomness(3, variation=0.15)
    print(f"\nOriginal vs Randomized:")
    original = manager.get_scenario(3)
    
    key_params = ['Throughput (Mbps)', 'Latency (ms)', 'SNR (dB)']
    for param in key_params:
        orig_val = original['parameters'][param]
        rand_val = randomized['parameters'][param]
        diff = ((rand_val - orig_val) / orig_val) * 100
        print(f"  {param}:")
        print(f"    Original: {orig_val:.2f}")
        print(f"    Randomized: {rand_val:.2f} ({diff:+.1f}%)")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)