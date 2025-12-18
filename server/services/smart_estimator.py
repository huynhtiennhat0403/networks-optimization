"""
Smart Estimator for Mode 3 (Simple Input)
Updated to match 'generate_dataset.py' logic strictly.
Removes unused metrics (SNR, BER, PDR, Modulation).
"""

import logging
from typing import Dict
import numpy as np # Dùng numpy nếu server có cài, hoặc dùng math chuẩn cũng được

logger = logging.getLogger(__name__)

class SmartEstimator:
    """
    Estimates missing PHYSICAL parameters based on user inputs.
    Target output matches 'generate_dataset.py' structure.
    """
    
    def __init__(self):
        logger.info("🤖 SmartEstimator initialized (Physics-based logic)")
        
        # Mapping Congestion Text -> Score (để logic nội bộ dùng nếu cần)
        self.congestion_map = {'Low': 1, 'Medium': 2, 'High': 3}

    def estimate(self, simple_input: Dict) -> Dict:
        """
        Input: user_speed, battery_level, signal_strength, latency, throughput
        Output: Full dict matching 'generate_dataset.py' columns
        """
        
        # 1. Validate & Extract Input
        self._validate_input(simple_input)
        
        user_speed_kmh = simple_input['user_speed']
        battery = simple_input['battery_level']
        signal = simple_input['signal_strength']
        latency = simple_input['latency']
        throughput = simple_input['throughput']
        
        # Context (Optional)
        location = simple_input.get('location', 'home').lower()
        user_activity = simple_input.get('user_activity', 'browsing').lower()

        # Convert speed km/h -> m/s
        user_speed_ms = user_speed_kmh / 3.6

        # --- HEURISTIC LOGIC (Mô phỏng ngược generate_dataset.py) ---

        # 1. Network Congestion (Low/Medium/High)
        # Logic: Latency cao hoặc Throughput thấp -> High Congestion
        congestion = self._estimate_congestion(latency, throughput, user_activity)
        
        # 2. Distance from Base Station (m)
        # Logic gốc: 1000 - (sig_norm * 900)
        # Ta đảo ngược: Signal càng yếu -> Distance càng xa
        # Signal: -120 (xa) -> -50 (gần)
        sig_norm = (signal - (-120)) / (-50 - (-120)) # 0..1
        distance = 1000 - (sig_norm * 900)
        # Thêm chút sai số ngẫu nhiên cho tự nhiên (không bắt buộc)
        distance = max(10.0, distance)

        # 3. Handover Events
        # Logic gốc: Dựa vào Speed thresholds
        handovers = 0
        if user_speed_kmh < 10: handovers = 0
        elif user_speed_kmh < 40: handovers = 1
        elif user_speed_kmh < 80: handovers = 2
        else: handovers = 3
        # Nếu ở outdoor/vehicle thì có thể tăng thêm
        if location in ['vehicle', 'outdoor'] and handovers < 4:
            handovers += 1

        # 4. Transmission Power (dBm)
        # Logic gốc: tx_power = 23 + ((-90 - signal) * 0.5)
        # Signal yếu -> Cần phát mạnh hơn
        tx_power = 23 + ((-90 - signal) * 0.5)
        tx_power = max(5.0, min(30.0, tx_power))

        # 5. Power Consumption (mW)
        # Logic gốc: Base 500. Signal < -90 (+200). Battery < 20 (-100).
        base_power = 500.0
        if signal < -90: base_power += 200
        if battery < 20: base_power -= 100
        # Thêm yếu tố throughput (tải nặng tốn pin)
        if throughput > 50: base_power += 50
        
        power_consumption = base_power
        
        # --- PACKING RESULT ---
        # Chỉ trả về các cột mà ModelWrapper (và processing_data) cần
        estimated_params = {
            'User Speed (m/s)': round(user_speed_ms, 2),
            'Signal Strength (dBm)': float(signal),
            'Battery Level (%)': int(battery),
            'Network Congestion': congestion, # ModelWrapper sẽ tự map sang Score (1,2,3)
            'Distance from Base Station (m)': round(distance, 2),
            'Handover Events': int(handovers),
            'Power Consumption (mW)': round(power_consumption, 2),
            'Transmission Power (dBm)': round(tx_power, 2),
        }
        
        logger.info(f"✅ Estimated params: {estimated_params}")
        return estimated_params

    def _validate_input(self, simple_input: Dict) -> None:
        required = ['user_speed', 'battery_level', 'signal_strength', 'latency', 'throughput']
        for field in required:
            if field not in simple_input:
                raise ValueError(f"Missing required field: {field}")

    def _estimate_congestion(self, latency, throughput, activity):
        # Đơn giản hóa: Latency > 70ms hoặc Throughput < 10Mbps -> High
        score = 0
        if latency > 70: score += 1
        if throughput < 10: score += 1
        if activity in ['gaming', 'streaming'] and latency > 50: score += 1
        
        if score >= 2: return 'High'
        if score == 1: return 'Medium'
        return 'Low'