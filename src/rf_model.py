import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_final_model(
    train_path="data/synthetic/train_smote_balanced.csv", 
    model_dir="models",
    report_dir="reports"
):
    print("🚀 Bắt đầu huấn luyện mô hình Random Forest cuối cùng...")
    
    # --- 1️⃣ Load dữ liệu ---
    if not os.path.exists(train_path):
        print(f"❌ Lỗi: Không tìm thấy file {train_path}. Hãy chạy smote.py trước.")
        return

    df_train = pd.read_csv(train_path)
    print(f"📊 Dữ liệu training: {len(df_train)} mẫu (từ {train_path})")
    
    target_col = 'RF Link Quality'
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    feature_names = X_train.columns.tolist()
    
    # --- 2️⃣ Huấn luyện Model ---
    print("🤖 Đang train model...")
    # Tăng n_estimators lên 200 để model ổn định hơn
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_final.fit(X_train, y_train)
    
    # --- 3️⃣ Lưu Model ---
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_final_model.pkl")
    joblib.dump(rf_final, model_path)
    print(f"✅ Đã lưu model tại: {model_path}")
    
    # --- 4️⃣ Vẽ & Lưu Feature Importance ---
    print("📊 Đang vẽ biểu đồ độ quan trọng đặc trưng...")
    
    importances = rf_final.feature_importances_
    # Tạo DataFrame để dễ vẽ
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance (Final RF Model)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    os.makedirs(report_dir, exist_ok=True)
    plot_path = os.path.join(report_dir, "final_feature_importance.png")
    plt.savefig(plot_path)
    print(f"✅ Đã lưu biểu đồ tại: {plot_path}")
    
    # In top features ra màn hình
    print("\n🏆 Top 5 Features quan trọng nhất:")
    print(fi_df.head(5).to_string(index=False))

if __name__ == "__main__":
    # Đảm bảo đường dẫn đúng với file combined bạn đã tạo ở bước trước
    train_final_model(train_path="data/synthetic/train_smote_balanced.csv")