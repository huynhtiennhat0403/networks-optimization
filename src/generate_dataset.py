import pandas as pd
import numpy as np
import os

def generate_physics_data(num_samples=5000, output_path="data/raw/wireless_communication_dataset.csv"):
    print(f"🚀 Đang sinh {num_samples} mẫu dữ liệu chuẩn vật lý (Label dạng Chữ)...")
    
    data = []
    
    # Cấu hình tắc nghẽn
    congestion_levels = ['Low', 'Medium', 'High']
    congestion_map = {'Low': 1, 'Medium': 2, 'High': 3}
    
    for _ in range(num_samples):
        # --- A. SINH INPUT NGẪU NHIÊN ---
        
        # 1. Signal Strength (-120 đến -50 dBm)
        signal = np.random.normal(-85, 15)
        signal = np.clip(signal, -120, -50)
        
        # 2. User Speed (0 đến 100 km/h -> m/s)
        rand = np.random.random()
        if rand < 0.5: speed_kmh = np.random.uniform(0, 10)
        elif rand < 0.8: speed_kmh = np.random.uniform(30, 50)
        else: speed_kmh = np.random.uniform(60, 100)
        speed_ms = speed_kmh / 3.6
        
        # 3. Battery (10% đến 100%)
        battery = np.random.randint(10, 101)
        
        # 4. Congestion (Low/Med/High)
        congestion = np.random.choice(congestion_levels, p=[0.4, 0.4, 0.2])
        cong_score = congestion_map[congestion]
        
        # --- B. ƯỚC LƯỢNG THÔNG SỐ PHỤ ---
        
        # 5. Distance
        sig_norm = (signal - (-120)) / (-50 - (-120))
        distance = 1000 - (sig_norm * 900)
        distance += np.random.normal(0, 50)
        distance = max(10, distance)
        
        # 6. Handover
        if speed_kmh < 10: handovers = 0
        elif speed_kmh < 40: handovers = np.random.randint(0, 2)
        elif speed_kmh < 80: handovers = np.random.randint(1, 4)
        else: handovers = np.random.randint(2, 5)
        
        # 7. Power Consumption
        base_power = 500
        if signal < -90: base_power += 200
        if battery < 20: base_power -= 100
        power_consumption = base_power + np.random.normal(0, 50)
        
        # 8. Transmission Power
        tx_power = 23 + ((-90 - signal) * 0.5)
        tx_power = np.clip(tx_power, 5, 30)
        
        # --- C. TÍNH TOÁN KẾT QUẢ ---
        
        # Throughput Logic
        tp_base = 100 * sig_norm
        tp_cong_penalty = (cong_score - 1) * 25
        tp_speed_penalty = speed_ms * 1.5
        
        throughput = tp_base - tp_cong_penalty - tp_speed_penalty
        throughput += np.random.normal(0, 5)
        throughput = np.clip(throughput, 1, 150)
        
        # Latency Logic
        lat_base = 20 + (1 - sig_norm) * 50
        lat_cong_penalty = (cong_score - 1) * 40
        
        latency = lat_base + lat_cong_penalty
        latency += np.random.normal(0, 10)
        latency = np.clip(latency, 5, 500)
        
        # --- D. GÁN NHÃN (DẠNG CHỮ) ---
        quality = 'Poor'
        if throughput >= 40 and latency <= 50:
            quality = 'Good'
        elif throughput >= 15 and latency <= 100:
            quality = 'Moderate'
        else:
            quality = 'Poor'
            
        # --- E. ĐÓNG GÓI ---
        row = {
            'User Speed (m/s)': round(speed_ms, 2),
            'Signal Strength (dBm)': round(signal, 2),
            'Battery Level (%)': int(battery),
            'Network Congestion': congestion, # Chữ: Low/Med/High
            'Distance from Base Station (m)': round(distance, 2),
            'Handover Events': int(handovers),
            'Power Consumption (mW)': round(power_consumption, 2),
            'Transmission Power (dBm)': round(tx_power, 2),
            
            # Target Cols
            'Throughput (Mbps)': round(throughput, 2),
            'Latency (ms)': round(latency, 2),
            'RF Link Quality': quality # Chữ: Good/Moderate/Poor
        }
        data.append(row)

    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Đã lưu {len(df)} mẫu dữ liệu tại: {output_path}")
    print("📊 Phân phối nhãn:", df['RF Link Quality'].value_counts().to_dict())

if __name__ == "__main__":
    generate_physics_data()