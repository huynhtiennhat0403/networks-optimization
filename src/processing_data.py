import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

def process_data(input_path, output_folder="data/processed", model_dir='models'):
    """
    Xử lý dữ liệu: Map Target từ Chữ sang Số & Feature Engineering
    """
    
    # --- 1️⃣ Đọc dữ liệu ---
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"📊 Tổng số mẫu: {len(df)}")
    
    target_col = 'RF Link Quality'

    # --- 2️⃣ MAP TARGET (Chữ -> Số) ---
    print("🔄 Đang chuyển đổi nhãn sang dạng số...")
    quality_map = {'Poor': 0, 'Moderate': 1, 'Good': 2}
    
    # Map và xử lý lỗi nếu có giá trị lạ
    df[target_col] = df[target_col].map(quality_map)
    
    # Kiểm tra xem có dòng nào bị NaN (do lỗi chính tả trong file raw) không
    if df[target_col].isnull().any():
        print("⚠️ Cảnh báo: Có nhãn không hợp lệ, đang loại bỏ...")
        df.dropna(subset=[target_col], inplace=True)
        
    df[target_col] = df[target_col].astype(int)
    print(f"✅ Phân phối sau khi map: {df[target_col].value_counts().to_dict()}")

    # --- 3️⃣ Map Congestion (Chữ -> Số để tính toán) ---
    congestion_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Network Congestion Score'] = df['Network Congestion'].map(congestion_map).fillna(2).astype(int)

    # --- 4️⃣ Feature Engineering ---
    print("🛠️ Đang tạo các features mới...")
    
    df['Mobility_Impact'] = df['User Speed (m/s)'] * (df['Handover Events'] + 1)
    df['Signal_Quality_Index'] = df['Signal Strength (dBm)'] * df['Network Congestion Score']
    df['Device_Stress_Level'] = df['Power Consumption (mW)'] / (df['Battery Level (%)'] + 1)
    df['Log_Distance'] = np.log1p(df['Distance from Base Station (m)'])

    # --- 5️⃣ Lọc bỏ Columns ---
    # Bỏ Throughput, Latency (Đáp án) và các cột text gốc
    cols_to_drop = [
        'Throughput (Mbps)', 
        'Latency (ms)', 
        'Network Congestion', # Bỏ cột chữ, giữ cột Score
        target_col # Tách target riêng
    ]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    print(f"\n🔍 Features dùng để train ({len(X.columns)}): {list(X.columns)}")
    
    # --- 6️⃣ Scaling ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Gán lại target
    df_processed = X_processed.copy()
    df_processed[target_col] = y.values
    
    # --- 7️⃣ Chia Train/Test ---
    train_df, test_df = train_test_split(
        df_processed, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_processed[target_col]
    )
    
    # --- 8️⃣ Lưu kết quả ---
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)
    
    # Lưu metadata
    joblib.dump(scaler, os.path.join(model_dir, "minmax_scaler.pkl"))
    
    feature_info = {
        'numerical_features': list(X.columns),
        'categorical_features': [],
        'all_features': list(X.columns) + [target_col]
    }
    joblib.dump(feature_info, os.path.join(model_dir, "feature_info.pkl"))
    
    print(f"\n✅ Xử lý hoàn tất!")
    return train_df, test_df

if __name__ == "__main__":
    process_data("data/raw/wireless_communication_dataset.csv")