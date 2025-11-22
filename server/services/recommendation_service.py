# File: services/recommendation_service.py
import json
import os
import logging
from typing import Dict, Any, List, Optional

# KHÔNG CẦN import ModelWrapper hay Preprocessor nữa

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, base_dir: str):
        self.config = {}
        self.ranges = {} # Để lưu trữ min/max/mean

        # 1. Tải file recommendations.json 
        reco_path = os.path.join(base_dir, "server", "data", "recommendations", "recommendations.json")
        try:
            with open(reco_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info("✅ Tải thành công recommendations.json")
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải recommendations.json: {e}")

        # 2. Tải file network_ranges.json 
        ranges_path = os.path.join(base_dir, "config", "network_ranges.json")
        try:
            with open(ranges_path, 'r', encoding='utf-8') as f:
                self.ranges = json.load(f)
            logger.info("✅ Tải thành công network_ranges.json")
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải network_ranges.json: {e}")

    def _get_problem_id(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Dùng kiến thức từ ảnh Feature Importance
        để tìm ra VẤN ĐỀ (root cause)
        """
        
        # Ưu tiên 1: Signal Strength (Mean: -69.35)
        if params.get('Signal Strength (dBm)', -70) < -80: 
            return 'Signal Strength (dBm)'
            
        # Ưu tiên 2: PDR (%) (Mean: 75.0)
        if params.get('PDR (%)', 100) < 80: 
            return 'PDR (%)'
            
        # Ưu tiên 3: SNR (dB) (Mean: 17.32)
        if params.get('SNR (dB)', 99) < 15: 
            return 'SNR (dB)'
            
        # Ưu tiên 4: Handover Events (Mean: 1.98)
        if params.get('Handover Events', 0) >= 3:
            return 'Handover Events'
            
        # Ưu tiên 5: Latency (ms) (Mean: 50.77)
        if params.get('Latency (ms)', 0) > 70:
            return 'Latency (ms)'
            
            
        return None # Không tìm thấy vấn đề cụ thể

    def get_recommendation(self, params: Dict[str, Any], prediction_label: str) -> str:
        """
        Lấy lời khuyên dựa trên logic ưu tiên
        """
        
        # 1. Lấy thông báo mặc định trước
        default_messages = self.config.get("default_messages", {})
        recommendation = default_messages.get(prediction_label, "Không có lời khuyên.")
        
        # 2. Nếu kết quả Tốt, trả về luôn
        if prediction_label == "Good":
            return recommendation
            
        # 3. Nếu Poor/Moderate, tìm nguyên nhân gốc (root cause)
        try:
            problem_id = self._get_problem_id(params)
            
            # 4. Nếu tìm thấy vấn đề, lấy lời khuyên cụ thể từ JSON
            if problem_id:
                specific_recommendation = self.config.get("recommendations", {}).get(problem_id)
                if specific_recommendation:
                    logger.info(f"[Recommend] AI xác định vấn đề: {problem_id}")
                    # Kết hợp thông báo chung + lời khuyên cụ thể
                    return f"{recommendation} <strong>Nguyên nhân chính:</strong> {specific_recommendation}"
            
            # 5. Nếu không tìm thấy vấn đề cụ thể, trả về thông báo mặc định
            return recommendation

        except Exception as e:
            logger.error(f"Lỗi khi tạo recommendation: {e}")
            return default_messages.get(prediction_label, "Lỗi khi tạo lời khuyên.")