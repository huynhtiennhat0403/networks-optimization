"""
Smart Estimator for Mode 3
Estimates full 16 network parameters from simplified input (3-5 metrics)
Uses context-aware heuristics based on user activity, device type, and location
"""

import logging
from typing import Dict, Optional
import math

logger = logging.getLogger(__name__)


class SmartEstimator:
    """
    Intelligent parameter estimator for Mode 3 (Simple Input)
    
    Given 3-5 basic metrics + context, estimates all 20 network parameters
    using realistic heuristics based on wireless communication principles
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
            'video_call': {'bandwidth_intensity': 0.7, 'latency_sensitive': True, 'stability_importance': 0.95},
        }
        
        # Define device characteristics
        self.device_profiles = {
            'phone': {'power_consumption_factor': 0.8, 'antenna_efficiency': 0.9, 'cpu_power': 0.6},
            'laptop': {'power_consumption_factor': 1.2, 'antenna_efficiency': 0.95, 'cpu_power': 1.0},
            'tablet': {'power_consumption_factor': 1.0, 'antenna_efficiency': 0.92, 'cpu_power': 0.8},
            'iot': {'power_consumption_factor': 0.3, 'antenna_efficiency': 0.7, 'cpu_power': 0.2},
        }
    
    
    def estimate(self, simple_input: Dict) -> Dict:
        """
        Estimate full 20 network parameters from simplified input
        
        Args:
            simple_input: Dict with required keys (throughput, latency, signal_strength)
                         and optional context (user_activity, device_type, location, etc.)
        
        Returns:
            Dict with full 20 estimated parameters
        """
        
        # ==================== VALIDATE INPUT ====================
        try:
            self._validate_input(simple_input)
        except ValueError as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise
        
        # ==================== EXTRACT INPUTS ====================
        throughput = simple_input['throughput']
        latency = simple_input['latency']
        signal = simple_input['signal_strength']
        
        # Optional context (with defaults)
        user_activity = simple_input.get('user_activity', 'browsing').lower()
        device_type = simple_input.get('device_type', 'laptop').lower()
        location = simple_input.get('location', 'home').lower()
        connection_type = simple_input.get('connection_type', 'wifi').lower()
        battery_level = simple_input.get('battery_level', 100)
        network_congestion = simple_input.get('network_congestion', 1)
        
        logger.info(f"[ESTIMATE] Input: throughput={throughput}Mbps, latency={latency}ms, signal={signal}dBm")
        logger.info(f"[ESTIMATE] Context: activity={user_activity}, device={device_type}, location={location}")
        
        # ==================== HEURISTIC RULES ====================
        
        # 1. Signal Quality → SNR, BER, PDR, Modulation
        snr, ber, pdr, modulation = self._estimate_signal_quality(signal, throughput, latency)
        
        # 2. Throughput + Activity → Handovers, Retransmissions
        handovers, retransmissions = self._estimate_handovers_retransmissions(
            throughput, signal, user_activity, location
        )
        
        # 3. Device + Battery → Power Consumption
        power_consumption = self._estimate_power_consumption(device_type, battery_level, throughput, pdr)
        
        # 4. Signal + Congestion → Distance and other parameters
        distance = self._estimate_distance_from_signal(signal, connection_type)
        
        # 5. Activity + Context → Speed and Direction
        speed, direction = self._estimate_user_mobility(user_activity, location)
        
        # ==================== BUILD RESPONSE ====================
        estimated_params = {
            'User Speed (m/s)': speed,
            'User Direction (degrees)': direction,
            'Handover Events': handovers,
            'Distance from Base Station (m)': distance,
            'Signal Strength (dBm)': signal,
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
            'Transmission Power (dBm)': self._estimate_transmission_power(signal, snr),
            'Channel Bandwidth (MHz)': self._estimate_channel_bandwidth(throughput, snr),
            'MIMO Layers': self._estimate_mimo_layers(signal, throughput),
            'CQI': self._estimate_channel_quality_indicator(snr, pdr),
            'Spectral Efficiency (bits/s/Hz)': self._estimate_spectral_efficiency(throughput, snr),
        }
        
        logger.info(f"[ESTIMATE] Generated {len(estimated_params)} parameters")
        logger.info(f"[ESTIMATE] SNR={snr}dB, BER={ber:.4f}, PDR={pdr}%, Mod={modulation}")
        
        return estimated_params
    
    
    def _validate_input(self, simple_input: Dict) -> None:
        """
        Validate input parameters
        
        Raises:
            ValueError: If any required field is missing or out of range
        """
        required = ['throughput', 'latency', 'signal_strength']
        for field in required:
            if field not in simple_input:
                raise ValueError(f"Missing required field: {field}")
        
        throughput = simple_input['throughput']
        latency = simple_input['latency']
        signal = simple_input['signal_strength']
        
        if not (0 < throughput <= 300):
            raise ValueError(f"throughput must be 0-300 Mbps, got {throughput}")
        if not (0 < latency <= 500):
            raise ValueError(f"latency must be 0-500 ms, got {latency}")
        if not (-100 <= signal <= -40):
            raise ValueError(f"signal_strength must be -100 to -40 dBm, got {signal}")
        
        logger.debug(f"Input validation passed: throughput={throughput}, latency={latency}, signal={signal}")
    
    
    def _estimate_signal_quality(self, signal: float, throughput: float, latency: float) -> tuple:
        """
        Estimate SNR, BER, PDR, and Modulation Scheme based on signal strength and throughput
        
        Theory: Better signal → higher SNR → lower BER → higher PDR → higher modulation
        """
        # SNR estimation from signal strength (empirical formula)
        # Signal below -70 dBm typically gives SNR around 8-15 dB
        # Signal above -50 dBm typically gives SNR around 20-30 dB
        snr = max(5, min(30, -signal - 60))  # Simplified: SNR ≈ -Signal - 60
        
        # BER (Bit Error Rate) depends heavily on SNR
        # BER roughly follows: BER ≈ 0.5 * erfc(sqrt(2 * SNR))
        # For simplicity: BER ≈ 10^(-SNR/10) * constant
        ber = max(0.0001, min(0.1, 10 ** (-snr / 10) * 0.01))
        
        # PDR (Packet Delivery Ratio) - inverse relationship with BER
        # PDR ≈ 100 * (1 - BER)^N where N is packet size in bits
        # Assuming ~1500 byte packets = 12000 bits
        pdr = max(50, min(100, 100 * (1 - ber) ** 50))  # Simplified for 50-bit chunks
        
        # Modulation scheme selection based on SNR
        # Lower SNR → BPSK (robust, low data rate)
        # Medium SNR → QPSK (balanced)
        # Higher SNR → 16-QAM → 64-QAM (high data rate, less robust)
        if snr < 10:
            modulation = 'BPSK'
        elif snr < 15:
            modulation = 'QPSK'
        elif snr < 20:
            modulation = '16-QAM'
        else:
            modulation = '64-QAM'
        
        logger.debug(f"Signal quality: SNR={snr:.1f}dB, BER={ber:.6f}, PDR={pdr:.1f}%, Mod={modulation}")
        
        return snr, ber, pdr, modulation
    
    
    def _estimate_handovers_retransmissions(
        self, throughput: float, signal: float, activity: str, location: str
    ) -> tuple:
        """
        Estimate handover events and retransmission count based on signal and location
        """
        # Handovers depend on location mobility and signal stability
        location_handover_map = {
            'home': 0,
            'office': 0,
            'public': 1,
            'outdoor': 2,
            'vehicle': 3,
            'event': 2,
        }
        
        base_handovers = location_handover_map.get(location, 0)
        
        # Add variation based on signal strength (weaker signal = more handovers)
        if signal < -80:
            base_handovers += 2
        elif signal < -70:
            base_handovers += 1
        
        handovers = min(5, max(0, base_handovers))
        
        # Retransmissions: depend on throughput and signal quality
        # Lower throughput usually means more packet loss → more retransmissions
        # Weaker signal also increases retransmissions
        retransmission_base = max(1, 10 - throughput / 15)  # At 150Mbps, ~0 retransmissions
        
        if signal < -85:
            retransmission_base *= 2
        elif signal < -70:
            retransmission_base *= 1.5
        
        retransmissions = min(10, max(1, int(round(retransmission_base))))
        
        logger.debug(f"Handovers={handovers}, Retransmissions={retransmissions} (activity={activity}, location={location})")
        
        return handovers, retransmissions
    
    
    def _estimate_power_consumption(
        self, device_type: str, battery: float, throughput: float, pdr: float
    ) -> float:
        """
        Estimate power consumption based on device type and network activity
        
        Device power consumption ranges:
        - IoT: 50-200 mW
        - Phone: 200-800 mW
        - Tablet: 400-1200 mW
        - Laptop: 600-1800 mW
        """
        device_profile = self.device_profiles.get(device_type, self.device_profiles['laptop'])
        
        # Base power consumption by device
        base_power_map = {
            'iot': 100,
            'phone': 400,
            'tablet': 700,
            'laptop': 1000,
        }
        base_power = base_power_map.get(device_type, 800)
        
        # Increase power for higher throughput (more transmission power needed)
        throughput_factor = 1 + (throughput / 150) * 0.5  # +50% at 150 Mbps
        
        # Battery level affects power (lower battery = higher draw in some devices)
        battery_factor = 1 + (100 - battery) / 500  # +20% at 0% battery
        
        # PDR affects power (retransmissions = more power)
        retransmission_factor = max(1.0, 100 / pdr)
        
        power_consumption = base_power * throughput_factor * battery_factor * retransmission_factor
        power_consumption *= device_profile['power_consumption_factor']
        
        power_consumption = max(50, min(2000, power_consumption))  # Clamp to realistic range
        
        return round(power_consumption, 1)
    
    
    def _estimate_distance_from_signal(self, signal: float, connection_type: str) -> float:
        """
        Estimate distance from Base Station using Friis path loss formula
        
        Path Loss (dB) = Pt - Pr = 20*log10(4*pi*d*f/c) + Gt + Gr
        Simplified: Signal_dBm ≈ -100 - 20*log10(d/100) for typical WiFi/4G
        """
        # Reference: at 100m, signal is typically around -100 dBm for WiFi
        # This is rough; actual values depend on frequency, antenna gain, etc.
        
        if connection_type.lower() in ['4g', '5g']:
            reference_signal_at_100m = -90  # Better propagation for cellular
        else:  # wifi, ethernet
            reference_signal_at_100m = -100
        
        # Friis formula inverted: d ≈ 10^((ref_signal - current_signal) / 20) * 100
        distance = 10 ** ((reference_signal_at_100m - signal) / 20) * 100
        
        # Clamp to reasonable range (10m to 2000m)
        distance = max(10, min(2000, distance))
        
        return round(distance, 1)
    
    
    def _estimate_user_mobility(self, activity: str, location: str) -> tuple:
        """
        Estimate user speed and direction based on activity and location
        
        Returns: (speed_m/s, direction_degrees)
        """
        # Activity-based speed estimates
        activity_speed_map = {
            'browsing': (0.1, 0),      # Static or minimal movement
            'streaming': (0.2, 0),      # Usually sitting
            'gaming': (0.1, 0),         # Usually sitting
            'downloading': (0.1, 0),    # Usually sitting
            'video_call': (0.2, 0),     # Usually sitting
        }
        
        speed, direction = activity_speed_map.get(activity, (0.1, 0))
        
        # Location affects mobility
        if location == 'vehicle':
            speed = 8 + speed  # Add ~8 m/s (30 km/h) for vehicle
            direction = 90  # Assume moving east
        elif location == 'outdoor':
            speed += 1  # Add walking speed
            direction = 45
        
        speed = min(30, speed)  # Cap at realistic max
        direction = direction % 360
        
        return round(speed, 1), round(direction, 1)
    
    
    def _estimate_transmission_power(self, signal: float, snr: float) -> float:
        """
        Estimate transmission power based on signal quality
        
        Typical range: 10-30 dBm for WiFi/4G
        Weaker signals need more power to maintain connection
        """
        if snr < 10:
            tx_power = 28  # Maximum power for weak signals
        elif snr < 15:
            tx_power = 23
        elif snr < 20:
            tx_power = 18
        else:
            tx_power = 13
        
        return float(tx_power)
    
    
    def _estimate_channel_bandwidth(self, throughput: float, snr: float) -> float:
        """
        Estimate channel bandwidth using Shannon capacity
        
        C = B * log2(1 + SNR)
        B ≈ C / log2(1 + SNR)
        """
        snr_linear = 10 ** (snr / 10)
        capacity = throughput * (10 ** 6)  # Convert Mbps to bps
        
        # Shannon capacity: B ≈ C / log2(1 + SNR)
        if snr_linear > 0:
            bandwidth = capacity / (math.log2(1 + snr_linear) * 1e6)
            bandwidth = max(5, min(160, bandwidth))  # 5-160 MHz realistic
        else:
            bandwidth = 20
        
        # Round to standard values: 5, 10, 20, 40, 80, 160 MHz
        standard_bw = [5, 10, 20, 40, 80, 160]
        bandwidth = min(standard_bw, key=lambda x: abs(x - bandwidth))
        
        return float(bandwidth)
    
    
    def _estimate_mimo_layers(self, signal: float, throughput: float) -> int:
        """
        Estimate number of MIMO layers based on signal quality and throughput
        
        Better signal and higher throughput support more MIMO layers
        """
        if throughput < 20:
            mimo_layers = 1
        elif throughput < 50:
            mimo_layers = 2
        elif throughput < 100:
            mimo_layers = 4
        else:
            mimo_layers = 8
        
        # Reduce MIMO if signal is weak
        if signal < -85:
            mimo_layers = max(1, mimo_layers // 2)
        
        return mimo_layers
    
    
    def _estimate_channel_quality_indicator(self, snr: float, pdr: float) -> int:
        """
        Estimate CQI (Channel Quality Indicator) - ranges from 0 to 15
        
        Based on SNR and packet delivery ratio
        """
        # CQI mapping (simplified):
        # CQI 0-4: Very poor channel
        # CQI 5-8: Poor to moderate channel
        # CQI 9-12: Good channel
        # CQI 13-15: Excellent channel
        
        if snr < 5:
            cqi = 2
        elif snr < 10:
            cqi = 5
        elif snr < 15:
            cqi = 9
        else:
            cqi = 13
        
        # Adjust for PDR
        if pdr < 80:
            cqi = max(0, cqi - 3)
        elif pdr > 95:
            cqi = min(15, cqi + 2)
        
        return cqi
    
    
    def _estimate_spectral_efficiency(self, throughput: float, snr: float) -> float:
        """
        Estimate Spectral Efficiency (bits/s/Hz)
        
        Formula: SE ≈ log2(1 + SNR)
        """
        snr_linear = 10 ** (snr / 10)
        
        if snr_linear > 0:
            se = math.log2(1 + snr_linear)
        else:
            se = 0.1
        
        se = max(0.1, min(10, se))  # Clamp to realistic range
        
        return round(se, 2)
    
    
    def validate_full_parameters(self, params: Dict) -> bool:
        """
        Validate that all 20 parameters are in valid ranges
        
        Args:
            params: Dict of estimated parameters
        
        Returns:
            bool: True if all parameters are valid
        
        Raises:
            ValueError: If any parameter is out of range
        """
        expected_params = {
            'User Speed (m/s)': (0, 30),
            'User Direction (degrees)': (0, 360),
            'Handover Events': (0, 5),
            'Distance from Base Station (m)': (10, 2000),
            'Signal Strength (dBm)': (-100, -40),
            'SNR (dB)': (5, 35),
            'BER': (0.0001, 0.1),
            'PDR (%)': (50, 100),
            'Network Congestion': (0, 2),
            'Throughput (Mbps)': (0, 300),
            'Latency (ms)': (0, 500),
            'Retransmission Count': (1, 10),
            'Power Consumption (mW)': (50, 2000),
            'Battery Level (%)': (0, 100),
            'Transmission Power (dBm)': (10, 30),
            'Channel Bandwidth (MHz)': (5, 160),
            'MIMO Layers': (1, 8),
            'CQI': (0, 15),
            'Spectral Efficiency (bits/s/Hz)': (0.1, 10),
        }
        
        for param_name, (min_val, max_val) in expected_params.items():
            if param_name not in params:
                raise ValueError(f"Missing parameter: {param_name}")
            
            value = params[param_name]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Parameter {param_name}={value} out of range [{min_val}, {max_val}]"
                )
        
        logger.info("✅ All parameters validated successfully")
        return True
