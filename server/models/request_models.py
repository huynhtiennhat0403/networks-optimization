"""
Pydantic Request Models
Định nghĩa schemas cho tất cả requests đến API
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Optional


class AutoPredictRequest(BaseModel):
    """
    Mode 1: Auto Collect
    Client tự động thu thập đầy đủ network metrics
    """
    metrics: Dict[str, float] = Field(
        ..., 
        description="Complete network metrics collected automatically",
        example={
            "Throughput (Mbps)": 45.5,
            "Latency (ms)": 25.3,
            "Signal Strength (dBm)": -65.0,
            "SNR (dB)": 22.5,
            "Packet Loss (%)": 1.2,
            "Jitter (ms)": 5.8,
            "Network Congestion": 1,
            "Bandwidth (MHz)": 40,
            "Frequency (GHz)": 5.0,
            "Hour": 14,
            "Day of Week": 2,
            "Device Type": 0,
            "User Activity": 1,
            "Location": 0,
            "Connection Type": 0,
            "Connected Devices": 5,
            "Modulation Scheme_16-QAM": 0,
            "Modulation Scheme_256-QAM": 1,
            "Modulation Scheme_64-QAM": 0,
            "Modulation Scheme_QPSK": 0
        }
    )

    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate that metrics dict is not empty"""
        if not v:
            raise ValueError("Metrics dictionary cannot be empty")
        return v


class ScenarioPredictRequest(BaseModel):
    """
    Mode 2: Scenario Selection
    Client chọn một scenario từ danh sách có sẵn
    """
    scenario_id: int = Field(
        ..., 
        ge=1, 
        description="ID of predefined scenario (1-based)",
        example=1
    )


class SimplePredictRequest(BaseModel):
    """
    Mode 3: Simplified Input
    Client chỉ cung cấp 3-5 thông số cơ bản, server tự estimate phần còn lại
    """
    # Required fields (3 core metrics)
    throughput: float = Field(
        ..., 
        ge=1, 
        le=100, 
        description="Network throughput in Mbps",
        example=45.5
    )
    
    latency: float = Field(
        ..., 
        ge=1, 
        le=100, 
        description="Network latency in milliseconds",
        example=25.0
    )
    
    signal_strength: float = Field(
        ..., 
        ge=-100, 
        le=-40, 
        description="WiFi signal strength in dBm",
        example=-65.0
    )
    
    # Optional context fields
    user_activity: Optional[str] = Field(
        default="browsing",
        description="Current user activity: browsing, streaming, gaming, downloading",
        example="streaming"
    )
    
    device_type: Optional[str] = Field(
        default="laptop",
        description="Device type: phone, laptop, tablet, iot",
        example="laptop"
    )
    
    location: Optional[str] = Field(
        default="home",
        description="User location: home, office, public, outdoor",
        example="home"
    )
    
    connection_type: Optional[str] = Field(
        default="wifi",
        description="Connection type: wifi, 4g, 5g, ethernet",
        example="wifi"
    )

    @validator('user_activity')
    def validate_activity(cls, v):
        """Validate user activity"""
        valid = ['browsing', 'streaming', 'gaming', 'downloading']
        if v.lower() not in valid:
            raise ValueError(f"user_activity must be one of {valid}")
        return v.lower()
    
    @validator('device_type')
    def validate_device(cls, v):
        """Validate device type"""
        valid = ['phone', 'laptop', 'tablet', 'iot']
        if v.lower() not in valid:
            raise ValueError(f"device_type must be one of {valid}")
        return v.lower()
    
    @validator('location')
    def validate_location(cls, v):
        """Validate location"""
        valid = ['home', 'office', 'public', 'outdoor']
        if v.lower() not in valid:
            raise ValueError(f"location must be one of {valid}")
        return v.lower()
    
    @validator('connection_type')
    def validate_connection(cls, v):
        """Validate connection type"""
        valid = ['wifi', '4g', '5g', 'ethernet']
        if v.lower() not in valid:
            raise ValueError(f"connection_type must be one of {valid}")
        return v.lower()


class HealthCheckRequest(BaseModel):
    """Optional health check request with parameters"""
    detailed: bool = Field(
        default=False,
        description="Return detailed health information"
    )