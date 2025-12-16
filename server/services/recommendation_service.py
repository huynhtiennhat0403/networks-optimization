import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, base_dir: str):
        self.config = {}
        
        # Đường dẫn file json
        reco_path = os.path.join(base_dir, "server", "data", "recommendations", "recommendations.json")
        
        try:
            if os.path.exists(reco_path):
                with open(reco_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info("✅ RecommendationService: Loaded recommendations.json")
            else:
                logger.warning(f"⚠️ RecommendationService: File not found at {reco_path}")
        except Exception as e:
            logger.error(f"❌ Error loading recommendations: {e}")

    def _get_problem_id(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Xác định nguyên nhân gốc rễ (Root Cause Analysis)
        Dựa trên Feature Importance: Congestion > Signal > Speed > Distance
        """
        
        # 1. Network Congestion (Quan trọng nhất theo biểu đồ)
        # Giá trị có thể là 'High' (text) hoặc 3 (số) tùy vào thời điểm gọi
        cong = params.get('Network Congestion')
        if cong == 'High' or cong == 3:
            return 'Network Congestion'

        # 2. Signal Strength (Quan trọng nhì)
        # Ngưỡng -90dBm là rất yếu
        sig = params.get('Signal Strength (dBm)', -70)
        if sig < -90:
            return 'Signal Strength (dBm)'

        # 3. User Speed (m/s)
        # 15 m/s ~ 54 km/h
        speed = params.get('User Speed (m/s)', 0)
        if speed > 15:
            return 'User Speed (m/s)'

        # 4. Distance from Base Station (m)
        dist = params.get('Distance from Base Station (m)', 0)
        if dist > 800:
            return 'Distance from Base Station (m)'

        # 5. Battery Level (%) - Ít quan trọng với model nhưng dễ sửa với User
        batt = params.get('Battery Level (%)', 100)
        if batt < 20:
            return 'Battery Level (%)'
            
        return None

    def get_recommendation(self, params: Dict[str, Any], prediction_label: str) -> str:
        """
        Tạo lời khuyên dựa trên nhãn dự đoán và thông số đầu vào
        """
        # Lấy thông báo chung (Default message)
        default_msgs = self.config.get("default_messages", {})
        base_msg = default_msgs.get(prediction_label, "Chất lượng mạng chưa xác định.")
        
        # Nếu mạng Tốt, không cần khuyên gì thêm
        if prediction_label == "Good":
            return base_msg

        # Nếu mạng Kém/Trung bình, tìm nguyên nhân
        problem_id = self._get_problem_id(params)
        
        if problem_id:
            advice_map = self.config.get("recommendations", {})
            advice = advice_map.get(problem_id)
            
            if advice:
                # Format HTML để hiển thị đẹp trên React
                return f"{base_msg} <br/><br/>👉 <b>Lời khuyên:</b> {advice}"
        
        return base_msg

# Test nhanh (Optional)
if __name__ == "__main__":
    # Giả lập đường dẫn để test
    service = RecommendationService(base_dir=".")
    sample_params = {
        "Network Congestion": "High",
        "Signal Strength (dBm)": -95,
        "User Speed (m/s)": 5
    }
    print(service.get_recommendation(sample_params, "Poor"))