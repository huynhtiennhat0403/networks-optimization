"""
Pydantic Request Models
Định nghĩa schemas cho tất cả requests đến API
"""

from pydantic import BaseModel, Field, field_validator as validator
from typing import Optional

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
    Client cung cấp 5 thông số user input + optional contexts
    Server tự estimate phần còn lại
    """

    user_speed: float = Field(..., ge=0, le=120)  # km/h
    
    battery_level: int = Field(..., ge=5, le=100)
    
    signal_strength: float = Field(
        ..., ge=-120, le=-40, 
        description="Signal strength in dBm (estimated from signal bars: 1-4)",
        example=-75.0
    )

    latency: float = Field(
        ..., 
        ge=0, 
        description="Network latency in ms",
        example=45.0
    )

    throughput: float = Field(
        ..., 
        ge=0, 
        description="Network throughput in Mbps",
        example=50.0
    )
    
    # Optional context fields
    user_activity: Optional[str] = Field(
        default="browsing",
        description="Current user activity: browsing, streaming, gaming, downloading",
        example="streaming"
    )
    
    device_type: Optional[str] = Field(
        default="laptop",
        description="Device type: phone, laptop, tablet",
        example="laptop"
    )
    
    location: Optional[str] = Field(
        default="home",
        description="User location: home, office, event, outdoor",
        example="home"
    )
    
    connection_type: Optional[str] = Field(
        default="4g",
        description="Connection type: 4g, 5g",
        example="4g"
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
        valid = ['phone', 'laptop', 'tablet']
        if v.lower() not in valid:
            raise ValueError(f"device_type must be one of {valid}")
        return v.lower()
    
    @validator('location')
    def validate_location(cls, v):
        """Validate location"""
        valid = ['home', 'office', 'event', 'outdoor', 'vehicle']
        if v.lower() not in valid:
            raise ValueError(f"location must be one of {valid}")
        return v.lower()
    
    @validator('connection_type')
    def validate_connection(cls, v):
        """Validate connection type"""
        valid = ['4g', '5g']
        if v.lower() not in valid:
            raise ValueError(f"connection_type must be one of {valid}")
        return v.lower()


class HealthCheckRequest(BaseModel):
    """Optional health check request with parameters"""
    detailed: bool = Field(
        default=False,
        description="Return detailed health information"
    )