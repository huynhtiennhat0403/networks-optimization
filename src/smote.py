import pandas as pd
import os
import joblib
import numpy as np
from imblearn.over_sampling import SMOTENC
import warnings
warnings.filterwarnings('ignore')

def apply_smote_nc(train_path, output_folder="data/synthetic", model_dir='models'):
    """
    Áp dụng SMOTE-NC và lưu vào folder synthetic
    Chiến lược: Cân bằng tất cả các lớp bằng với lớp chiếm đa số (Majority Class)
    """
    
    # --- 1️⃣ Đọc dữ liệu ---
    train_df = pd.read_csv(train_path)
    print(f"📊 Đọc dữ liệu Train từ: {train_path} ({len(train_df)} mẫu)")
    
    feature_info = joblib.load(os.path.join(model_dir, "feature_info.pkl"))
    target_col = 'RF Link Quality'
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    # --- 2️⃣ Xác định Categorical Indices ---
    categorical_features = feature_info['categorical_features']
    all_features = list(X_train.columns)
    categorical_indices = [all_features.index(col) for col in categorical_features if col in all_features]
    
    print(f"\n🔍 Features định danh (Indices: {categorical_indices}):")
    print(f"  {categorical_features}")
    
    # --- 3️⃣ Thiết lập Strategy thủ công (Bỏ Auto) ---
    # Đếm số lượng các lớp hiện tại
    class_counts = y_train.value_counts().to_dict()
    max_samples = max(class_counts.values()) # Lấy số lượng mẫu của lớp nhiều nhất (Poor)
    
    # Tạo dictionary strategy: Tất cả các lớp đều sẽ có số mẫu bằng max_samples
    sampling_strategy = {k: max_samples for k in class_counts.keys()}
    
    print(f"\n🎯 Chiến lược Sampling (Custom):")
    print(f"  - Phân phối gốc: {dict(sorted(class_counts.items()))}")
    print(f"  - Target Strategy: {dict(sorted(sampling_strategy.items()))}")
    print(f"  => Đưa tất cả các lớp về: {max_samples} mẫu")

    # --- 4️⃣ Áp dụng SMOTE-NC ---
    print(f"\n🔄 Đang chạy SMOTE-NC...")
    smote_nc = SMOTENC(
        categorical_features=categorical_indices,
        random_state=42,
        k_neighbors=5,
        sampling_strategy=sampling_strategy  # Sử dụng strategy thủ công
    )
    
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
    
    # --- 5️⃣ Lưu kết quả ---
    train_resampled_df = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    train_resampled_df[target_col] = y_train_resampled.values
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Tên file output
    output_filename = "train_smote_balanced.csv"
    output_path = os.path.join(output_folder, output_filename)
    
    train_resampled_df.to_csv(output_path, index=False)
    
    # Lưu SMOTE object
    joblib.dump(smote_nc, os.path.join(model_dir, "smote_nc_model.pkl"))
    
    print(f"\n✅ SMOTE hoàn tất!")
    print(f"📁 Dữ liệu Synthetic đã lưu tại: {output_path}")
    print(f"📈 Phân phối lớp mới: {train_resampled_df[target_col].value_counts().to_dict()}")

if __name__ == "__main__":
    # Input path trỏ tới file train.csv trong processed (do processing_data.py tạo ra)
    apply_smote_nc(train_path="data/processed/train.csv")