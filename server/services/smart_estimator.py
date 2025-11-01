"""
Smart Estimator for Mode 1
Estimates full network parameters from simplified input (5 metrics + contexts)
Uses context-aware heuristics based on user activity, device type, and location
"""

import logging
from typing import Dict, Optional
import math

logger = logging.getLogger(__name__)


class SmartEstimator:
    """
    Intelligent parameter estimator for Mode 3 (Simple Input)
    
    Given 5 basic metrics (user_speed, battery_level, signal_strength, latency, throughput)
    + contexts, estimates all required network parameters using realistic heuristics
    """
    
    def __init__(self):
        """Initialize the smart estimator with default configurations"""
        logger.info("🤖 SmartEstimator initialized with context-aware heuristics")
        
        # Define activity profiles (how intensive each activity is)
        self.activity_profiles = {
            'browsing': {'bandwidth_intensity': 0.3, 'latency_sensitive': False, 'stability_importance': 0.5},
            'streaming': {'bandwidth_intensity': 0.8, 'latency_sensitive': False, 'stability_importance': 0.8},
            'gaming': {'bandwidth_intensity': 0.6, 'latency_sensitive': True, 'stability_importance': 0.9},
            'downloading': {'bandwidth_intensity': 1.0, 'latency_sensitive': False, 'stability_importance': 0.6},
        }
        
        # Define device characteristics (removed 'iot')
        self.device_profiles = {
            'phone': {'power_consumption_factor': 0.8, 'antenna_efficiency': 0.9, 'cpu_power': 0.6},
            'laptop': {'power_consumption_factor': 1.2, 'antenna_efficiency': 0.95, 'cpu_power': 1.0},
            'tablet': {'power_consumption_factor': 1.0, 'antenna_efficiency': 0.92, 'cpu_power': 0.8},
        }
        
        # Network congestion mapping
        self.congestion_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    
    def estimate(self, simple_input: Dict) -> Dict:
        """
        Estimate full network parameters from simplified input
        
        Args:
            simple_input: Dict with required keys:
                - user_speed: float (km/h)
                - battery_level: int (%)
                - signal_strength: float (dBm)
                - latency: float (ms) - user input
                - throughput: float (Mbps) - user input
                - user_activity: str (optional)
                - device_type: str (optional)
                - location: str (optional)
                - connection_type: str (optional)
        
        Returns:
            Dict with full estimated parameters ready for model prediction
        """
        
        # ==================== VALIDATE INPUT ====================
        try:
            self._validate_input(simple_input)
        except ValueError as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise
        
        # ==================== EXTRACT INPUTS ====================
        user_speed_kmh = simple_input['user_speed']
        battery_level = simple_input['battery_level']
        signal_strength = simple_input['signal_strength']
        latency = simple_input['latency']
        throughput = simple_input['throughput']
        
        # Convert user_speed from km/h to m/s for internal calculations
        user_speed_ms = user_speed_kmh / 3.6
        
        # Optional context (with defaults)
        user_activity = simple_input.get('user_activity', 'browsing').lower()
        device_type = simple_input.get('device_type', 'laptop').lower()
        location = simple_input.get('location', 'home').lower()
        connection_type = simple_input.get('connection_type', '4g').lower()
        
        logger.info(f"[ESTIMATE] Input - speed: {user_speed_kmh}km/h, battery: {battery_level}%, signal: {signal_strength}dBm")
        logger.info(f"[ESTIMATE] User inputs - latency: {latency}ms, throughput: {throughput}Mbps")
        logger.info(f"[ESTIMATE] Context - activity: {user_activity}, device: {device_type}, location: {location}")
        
        # ==================== HEURISTIC RULES ====================
        
        # 1. Signal Quality → SNR, BER, PDR, Modulation
        snr, ber, pdr, modulation = self._estimate_signal_quality(
            signal_strength, throughput, latency
        )
        
        # 2. User mobility → Direction, Handovers
        direction, handovers = self._estimate_mobility_params(
            user_speed_ms, location, signal_strength
        )
        
        # 3. Connection quality → Retransmissions
        retransmissions = self._estimate_retransmissions(
            throughput, signal_strength, pdr
        )
        
        # 4. Device + Battery → Power Consumption
        power_consumption = self._estimate_power_consumption(
            device_type, battery_level, throughput, pdr
        )
        
        # 5. Signal + Connection Type → Distance
        distance = self._estimate_distance_from_signal(signal_strength, connection_type)
        
        # 6. Network congestion estimation
        network_congestion = self._estimate_network_congestion(
            latency, throughput, location
        )
        
        # 7. Advanced network parameters
        transmission_power = self._estimate_transmission_power(signal_strength, snr)
        
        # ==================== BUILD RESPONSE ====================
        estimated_params = {
            'User Speed (m/s)': round(user_speed_ms, 2),
            'User Direction (degrees)': direction,
            'Handover Events': handovers,
            'Distance from Base Station (m)': distance,
            'Signal Strength (dBm)': signal_strength,
            'SNR (dB)': snr,
            'BER': ber,
            'Modulation Scheme': modulation,
            'PDR (%)': pdr,
            'Network Congestion': network_congestion,
            'Throughput (Mbps)': throughput,
            'Latency (ms)': latency,
            'Retransmission Count': retransmissions,
            'Power Consumption (mW)': power_consumption,
            'Battery Level (%)': battery_level,
            'Transmission Power (dBm)': transmission_power,
        }
        
        logger.info(f"[ESTIMATE] Generated {len(estimated_params)} parameters")
        logger.info(f"[ESTIMATE] SNR={snr}dB, BER={ber:.6f}, PDR={pdr:.1f}%, Mod={modulation}")
        
        return estimated_params
    
    
    def _validate_input(self, simple_input: Dict) -> None:
        """
        Validate input parameters
        
        Raises:
            ValueError: If any required field is missing or out of range
        """
        required = ['user_speed', 'battery_level', 'signal_strength', 'latency', 'throughput']
        for field in required:
            if field not in simple_input:
                raise ValueError(f"Missing required field: {field}")
        
        user_speed = simple_input['user_speed']
        battery_level = simple_input['battery_level']
        signal_strength = simple_input['signal_strength']
        latency = simple_input['latency']
        throughput = simple_input['throughput']
        
        if not (0 <= user_speed <= 120):
            raise ValueError(f"user_speed must be 0-120 km/h, got {user_speed}")
        if not (0 <= battery_level <= 100):
            raise ValueError(f"battery_level must be 0-100%, got {battery_level}")
        if not (-120 <= signal_strength <= -50):
            raise ValueError(f"signal_strength must be -120 to -50 dBm, got {signal_strength}")
        if not (1 <= latency <= 1000):
            raise ValueError(f"latency must be 1-1000 ms, got {latency}")
        if not (1 <= throughput <= 1000):
            raise ValueError(f"throughput must be 1-1000 Mbps, got {throughput}")
        
        logger.debug(f"Input validation passed")
    
    
    def _estimate_signal_quality(self, signal: float, throughput: float, latency: float) -> tuple:
        """
        Estimate SNR, BER, PDR, and Modulation Scheme based on signal strength and throughput
        
        Based on dataset ranges:
        - SNR: 5.0 - 30.0 dB
        - BER: 0.0001 - 0.05
        - PDR: 50.0 - 100.0%
        - Modulation: BPSK, QPSK, 16-QAM, 64-QAM
        """
        # SNR estimation from signal strength
        # Map signal range [-100, -40] to SNR range [5, 30]
        snr = 5 + ((-signal + 100) / 60) * 25
        snr = max(5.0, min(30.0, snr))
        
        # BER (Bit Error Rate) - inverse relationship with SNR
        # Higher SNR = lower BER
        ber = 0.05 * (1 - (snr - 5) / 25)
        ber = max(0.0001, min(0.05, ber))
        
        # PDR (Packet Delivery Ratio) - inverse relationship with BER
        # Lower BER = higher PDR
        pdr = 50 + (1 - ber / 0.05) * 50
        pdr = max(50.0, min(100.0, pdr))
        
        # Modulation scheme selection based on SNR
        # Dataset has: BPSK, QPSK, 16-QAM, 64-QAM (no 256-QAM)
        if snr < 10:
            modulation = 'BPSK'
        elif snr < 15:
            modulation = 'QPSK'
        elif snr < 22:
            modulation = '16-QAM'
        else:
            modulation = '64-QAM'
        
        logger.debug(f"Signal quality: SNR={snr:.2f}dB, BER={ber:.6f}, PDR={pdr:.2f}%, Mod={modulation}")
        
        return round(snr, 2), round(ber, 6), round(pdr, 2), modulation
    
    
    def _estimate_mobility_params(self, speed_ms: float, location: str, signal: float) -> tuple:
        """
        Estimate user direction and handover events based on speed and location
        
        Based on dataset ranges:
        - Direction: 0-360 degrees
        - Handover Events: 0-4
        """
        # Direction estimation based on speed
        # Dataset range: 0.019 - 359.82 degrees
        if speed_ms < 0.5:  # Nearly stationary
            direction = 0.0
        elif speed_ms < 5:  # Walking
            direction = 90.0  # East
        elif speed_ms < 15:  # Running/cycling
            direction = 180.0  # South
        else:  # Vehicle
            direction = 270.0  # West
        
        # Add some variation based on location
        location_direction_offset = {
            'home': 0,
            'office': 45,
            'event': 90,
            'outdoor': 135,
            'vehicle': 180
        }
        direction = (direction + location_direction_offset.get(location, 0)) % 360
        
        # Handovers depend on location mobility and signal stability
        # Dataset range: 0-4
        location_handover_map = {
            'home': 0,
            'office': 1,
            'event': 2,
            'outdoor': 3,
            'vehicle': 4 
        }   
        
        base_handovers = location_handover_map.get(location, 1)
        
        # Add variation based on speed
        if speed_ms > 15:  # High speed vehicle
            base_handovers = min(4, base_handovers + 2)
        elif speed_ms > 8:  # Medium speed
            base_handovers = min(4, base_handovers + 1)
        
        # Weak signal increases handovers
        if signal < -90:
            base_handovers = min(4, base_handovers + 1)
        
        handovers = max(0, min(4, base_handovers))
        
        logger.debug(f"Mobility: direction={direction}°, handovers={handovers} (speed={speed_ms:.1f}m/s)")
        
        return round(direction, 2), handovers
    
    
    def _estimate_retransmissions(self, throughput: float, signal: float, pdr: float) -> int:
        """
        Estimate retransmission count
        
        Based on dataset range: 0-9
        """
        # Base retransmissions from throughput and PDR
        # Lower throughput and lower PDR = more retransmissions
        retransmission_base = 9 * (1 - throughput / 100) * (1 - pdr / 100)
        
        # Adjust for signal strength
        if signal < -90:
            retransmission_base *= 1.5
        elif signal < -80:
            retransmission_base *= 1.2
        
        retransmissions = max(0, min(9, int(round(retransmission_base))))
        
        logger.debug(f"Retransmissions={retransmissions} (throughput={throughput}, signal={signal}, pdr={pdr})")
        
        return retransmissions
    
    
    def _estimate_power_consumption(
        self, device_type: str, battery: float, throughput: float, pdr: float
    ) -> float:
        """
        Estimate power consumption based on device type and network activity
        
        Based on dataset range: 100.24 - 999.72 mW
        """
        device_profile = self.device_profiles.get(device_type, self.device_profiles['laptop'])
        
        # Base power consumption by device
        base_power_map = {
            'phone': 300,
            'tablet': 500,
            'laptop': 700,
        }
        base_power = base_power_map.get(device_type, 550)
        
        # Increase power for higher throughput
        throughput_factor = 1 + (throughput / 100) * 0.5
        
        # Lower battery = slightly higher power consumption
        battery_factor = 1 + (100 - battery) / 500
        
        # Lower PDR = more retransmissions = more power
        retransmission_factor = 1 + (100 - pdr) / 200
        
        power_consumption = base_power * throughput_factor * battery_factor * retransmission_factor
        power_consumption *= device_profile['power_consumption_factor']
        
        # Clamp to dataset range
        power_consumption = max(100.24, min(999.72, power_consumption))
        
        return round(power_consumption, 2)
    
    
    def _estimate_distance_from_signal(self, signal: float, connection_type: str) -> float:
        """
        Estimate distance from Base Station using path loss formula
        
        Based on dataset range: 10.25 - 999.93 m
        """
        # Path loss model: signal gets weaker with distance
        # Map signal range [-100, -40] to distance range [10, 1000]
        
        if connection_type in ['4g', '5g']:
            # Cellular has better range
            # -40 dBm = very close (~10m)
            # -100 dBm = far (~1000m)
            distance = 10 + ((-signal + 40) / 60) * 990
        else:
            # Other connections (if any future additions)
            distance = 10 + ((-signal + 40) / 60) * 990
        
        # Clamp to dataset range
        distance = max(10.25, min(999.93, distance))
        
        return round(distance, 2)
    
    
    def _estimate_network_congestion(self, latency: float, throughput: float, location: str) -> str:
        """
        Estimate network congestion level
        
        Based on dataset values: 'Low', 'Medium', 'High'
        Returns the string label directly
        """
        congestion_score = 0
        
        # High latency indicates congestion
        if latency > 70:
            congestion_score += 2
        elif latency > 40:
            congestion_score += 1
        
        # Low throughput indicates congestion
        if throughput < 30:
            congestion_score += 2
        elif throughput < 60:
            congestion_score += 1
        
        # Public locations tend to have more congestion
        location_congestion = {
            'home': 0,
            'office': 1,
            'event': 2, 
            'outdoor': 1,
            'vehicle': 1 
        }
        congestion_score += location_congestion.get(location, 0)
        
        # Map score to congestion label
        if congestion_score <= 2:
            return 0
        elif congestion_score <= 4:
            return 1
        else:
            return 2
    
    
    def _estimate_transmission_power(self, signal: float, snr: float) -> float:
        """
        Estimate transmission power based on signal quality
        
        Based on dataset range: 0.005 - 30.0 dBm
        """
        # Higher SNR and better signal = lower transmission power needed
        # Map SNR range [5, 30] to power range [30, 5]
        tx_power = 30 - ((snr - 5) / 25) * 25
        
        # Adjust based on signal strength
        if signal < -85:
            tx_power += 5  # Need more power for weak signal
        elif signal > -60:
            tx_power -= 5  # Can use less power for strong signal
        
        # Clamp to dataset range
        tx_power = max(0.005, min(30.0, tx_power))
        
        return round(tx_power, 2)