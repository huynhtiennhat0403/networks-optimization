import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

def analyze_and_test_leakage(
    train_path="data/processed/train.csv",
    test_path="data/processed/test.csv",
    output_dir="reports"
):
    print("🚀 Bắt đầu phân tích Feature Importance & Leakage Test...\n")
    
    # --- 1️⃣ Load dữ liệu ---
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    target_col = 'RF Link Quality'
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    feature_names = X_train.columns
    
    # --- 2️⃣ Train Random Forest ---
    print("🤖 Đang train Random Forest trên toàn bộ features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Đánh giá cơ bản
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy hiện tại (Full features): {acc:.2%}")
    
    # --- 3️⃣ Tính Feature Importance ---
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1] # Sắp xếp giảm dần
    
    print("\n📊 Top 5 Features quan trọng nhất:")
    for i in range(5):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
    # --- 4️⃣ Vẽ biểu đồ ---
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
    plt.title("Feature Importance (Độ quan trọng các đặc trưng)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'feature_importance_analysis.png')
    plt.savefig(plot_path)
    print(f"\n📈 Đã lưu biểu đồ tại: {plot_path}")
    
    # --- 5️⃣ LEAKAGE TEST (Thử nghiệm loại bỏ Top Features) ---
    print("\n" + "="*60)
    print("🧪 LEAKAGE TEST: Thử loại bỏ các features quan trọng nhất")
    print("="*60)
    
    # Lấy tên các features top đầu
    top_1_feature = feature_names[indices[0]]
    top_3_features = [feature_names[i] for i in indices[:3]]
    
    # Kịch bản 1: Bỏ Top 1 Feature
    print(f"\n🔻 Kịch bản 1: Loại bỏ Top 1 Feature ('{top_1_feature}')")
    X_train_drop1 = X_train.drop(columns=[top_1_feature])
    X_test_drop1 = X_test.drop(columns=[top_1_feature])
    
    rf_drop1 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_drop1.fit(X_train_drop1, y_train)
    acc_drop1 = accuracy_score(y_test, rf_drop1.predict(X_test_drop1))
    print(f"   => Accuracy mới: {acc_drop1:.2%} (Giảm {acc - acc_drop1:.2%})")
    
    # Kịch bản 2: Bỏ Top 3 Features
    print(f"\n🔻 Kịch bản 2: Loại bỏ Top 3 Features {top_3_features}")
    X_train_drop3 = X_train.drop(columns=top_3_features)
    X_test_drop3 = X_test.drop(columns=top_3_features)
    
    rf_drop3 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_drop3.fit(X_train_drop3, y_train)
    acc_drop3 = accuracy_score(y_test, rf_drop3.predict(X_test_drop3))
    print(f"   => Accuracy mới: {acc_drop3:.2%} (Giảm {acc - acc_drop3:.2%})")

    # --- Kết luận ---
    print("\n💡 KẾT LUẬN:")
    if acc_drop3 < 0.85: # Ngưỡng giả định
        print("   Có dấu hiệu của Data Leakage! Các features trên đang 'tiết lộ' trực tiếp kết quả.")
        print("   👉 Đề xuất: Loại bỏ các features này để bài toán thực tế hơn, sau đó mới dùng SMOTE.")
    else:
        print("   Các features còn lại vẫn đủ mạnh để dự đoán tốt.")

if __name__ == "__main__":
    analyze_and_test_leakage()