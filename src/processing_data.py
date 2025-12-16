import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def process_data(input_path, output_folder="data/processed", model_dir='models'):
    """
    Xử lý dữ liệu: 
    - Loại bỏ các cột gây Leakage (Signal Strength, SNR, PDR)
    - Encoding & Scaling
    - Chia Train/Test
    """
    
    # --- 1️⃣ Đọc dữ liệu ---
    df = pd.read_csv(input_path)
    print(f"📊 Tổng số mẫu ban đầu: {len(df)}")
    
    # --- 2️⃣ Vệ sinh dữ liệu ---
    target_col = 'RF Link Quality'
    df[target_col] = df[target_col].astype(str).str.strip()
    
    # Xóa các giá trị rác
    invalid_labels = ['0', 'nan', '', 'None']
    df = df[~df[target_col].isin(invalid_labels)].copy()
    
    # Map target
    rf_link_quality_map = {'Poor': 0, 'Moderate': 1, 'Good': 2}
    df[target_col] = df[target_col].map(rf_link_quality_map)
    df.dropna(subset=[target_col], inplace=True)
    df[target_col] = df[target_col].astype(int)
    
    # Map congestion
    congestion_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Network Congestion'] = df['Network Congestion'].astype(str).str.strip().map(congestion_map)
    df.dropna(subset=['Network Congestion'], inplace=True)
    df['Network Congestion'] = df['Network Congestion'].astype(int)
    
    df['Modulation Scheme'] = df['Modulation Scheme'].astype(str).str.strip()
    
    print(f"✅ Đã làm sạch dữ liệu. Còn lại {len(df)} mẫu.")

    # ==============================================================================
    # 🚨 QUAN TRỌNG: LOẠI BỎ CÁC CỘT GÂY DATA LEAKAGE 🚨
    # ==============================================================================
    leakage_cols = [
        # 'Signal Strength (dBm)', 
        'SNR (dB)',      # Khuyên bỏ: Vì SNR cao thì Quality chắc chắn tốt
        'BER',           # Khuyên bỏ: Bit Error Rate thấp thì Quality tốt
        'PDR (%)',       # Khuyên bỏ: Packet Delivery Ratio cao thì Quality tốt
        'Retransmission Count' # Khuyên bỏ: Số lần gửi lại liên quan trực tiếp đến lỗi mạng
    ]
    
    # Các cột có thể không quan trọng (Feature Selection - Optional)
    irrelevant_cols = ['User Direction (degrees)', 'Modulation Scheme'] # Hướng đi thường ít ảnh hưởng nếu Omni-directional antenna
    
    cols_to_remove = leakage_cols + irrelevant_cols
    
    print(f"\n✂️ Đang loại bỏ các cột Leakage & Không quan trọng: {cols_to_remove}")
    cols_to_drop = [col for col in cols_to_remove if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    # ==============================================================================
    
    # --- 4️⃣ Xác định feature types ---
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    categorical_features = []
    numerical_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or col in ['Network Congestion']:
            categorical_features.append(col)
        else:
            numerical_features.append(col)
            
    print(f"🔍 Features còn lại để train ({len(X.columns)}): {list(X.columns)}")
    
    # --- 5️⃣ Xử lý categorical features với Label Encoding ---
    label_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # --- 6️⃣ Chuẩn hóa numerical features ---
    scaler = MinMaxScaler()
    
    if numerical_features:
        X_scaled_num = scaler.fit_transform(X[numerical_features])
        X_processed = pd.DataFrame(X_scaled_num, columns=numerical_features, index=X.index)
        for col in categorical_features:
            X_processed[col] = X[col].values
    else:
        X_processed = X.copy()
    
    df_processed = X_processed.copy()
    df_processed[target_col] = y.values
    
    # --- 7️⃣ CHIA TRAIN/TEST ---
    train_df, test_df = train_test_split(
        df_processed, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_processed[target_col]
    )
    
    # --- 8️⃣ Lưu kết quả ---
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    train_path = os.path.join(output_folder, "train.csv")
    test_path = os.path.join(output_folder, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Lưu metadata (cập nhật lại danh sách feature mới)
    joblib.dump(scaler, os.path.join(model_dir, "minmax_scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))
    
    feature_info = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'target_mapping': rf_link_quality_map,
        'all_features': list(X_processed.columns) + [target_col]
    }
    joblib.dump(feature_info, os.path.join(model_dir, "feature_info.pkl"))
    
    print(f"\n✅ Xử lý hoàn tất (Đã loại bỏ Leakage)!")
    print(f"📁 Train set saved to: {train_path}")
    print(f"📁 Test set saved to: {test_path}")
    
    return train_df, test_df

if __name__ == "__main__":
    process_data("data/raw/wireless_communication_dataset.csv")