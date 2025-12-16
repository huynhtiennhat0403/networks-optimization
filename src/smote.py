import pandas as pd
import os
import joblib
import numpy as np
from imblearn.over_sampling import SMOTENC, SMOTE
import warnings
warnings.filterwarnings('ignore')

def apply_smote_nc(train_path, output_folder="data/synthetic", model_dir='models'):
    """
    Áp dụng SMOTE (hoặc SMOTE-NC) và lưu vào folder synthetic
    Tự động chuyển đổi giữa SMOTE thường và SMOTE-NC tùy vào dữ liệu.
    """
    
    # --- 1️⃣ Đọc dữ liệu ---
    if not os.path.exists(train_path):
        print(f"❌ Không tìm thấy file: {train_path}")
        return

    train_df = pd.read_csv(train_path)
    print(f"📊 Đọc dữ liệu Train từ: {train_path} ({len(train_df)} mẫu)")
    
    feature_info = joblib.load(os.path.join(model_dir, "feature_info.pkl"))
    target_col = 'RF Link Quality'
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    # --- 2️⃣ Xác định Categorical Indices ---
    categorical_features = feature_info.get('categorical_features', [])
    all_features = list(X_train.columns)
    
    # Tìm index của các cột categorical (nếu có)
    categorical_indices = [all_features.index(col) for col in categorical_features if col in all_features]
    
    print(f"\n🔍 Features định danh (Indices: {categorical_indices}):")
    print(f"  {categorical_features}")
    
    # --- 3️⃣ Thiết lập Strategy ---
    class_counts = y_train.value_counts().to_dict()
    max_samples = max(class_counts.values()) 
    sampling_strategy = {k: max_samples for k in class_counts.keys()}
    
    print(f"\n🎯 Chiến lược Sampling:")
    print(f"  - Phân phối gốc: {dict(sorted(class_counts.items()))}")
    print(f"  - Target: {dict(sorted(sampling_strategy.items()))}")

    # --- 4️⃣ Chọn thuật toán SMOTE phù hợp ---
    if len(categorical_indices) > 0:
        print(f"\n🔄 Phát hiện biến phân loại. Đang chạy SMOTE-NC...")
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            random_state=42,
            k_neighbors=5,
            sampling_strategy=sampling_strategy
        )
    else:
        print(f"\n🔄 Dữ liệu toàn bộ là số. Đang chạy SMOTE thường...")
        sampler = SMOTE(
            random_state=42,
            k_neighbors=5,
            sampling_strategy=sampling_strategy
        )
    
    # Thực hiện resample
    try:
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"❌ Lỗi khi chạy SMOTE: {str(e)}")
        # Fallback thử lại với k_neighbors nhỏ hơn nếu lỗi do ít dữ liệu
        print("⚠️ Thử lại với k_neighbors=1...")
        sampler.k_neighbors = 1
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
    # --- 5️⃣ Lưu kết quả ---
    train_resampled_df = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    train_resampled_df[target_col] = y_train_resampled.values
    
    os.makedirs(output_folder, exist_ok=True)
    
    output_filename = "train_smote_balanced.csv"
    output_path = os.path.join(output_folder, output_filename)
    
    train_resampled_df.to_csv(output_path, index=False)
    
    # Lưu model SMOTE để dùng lại nếu cần (dù thực tế ít khi dùng lại sampler)
    joblib.dump(sampler, os.path.join(model_dir, "smote_model.pkl"))
    
    print(f"\n✅ SMOTE hoàn tất!")
    print(f"📁 Dữ liệu Synthetic đã lưu tại: {output_path}")
    print(f"📈 Phân phối lớp mới: {train_resampled_df[target_col].value_counts().to_dict()}")

if __name__ == "__main__":
    apply_smote_nc(train_path="data/processed/train.csv")