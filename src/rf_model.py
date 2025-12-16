import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os

def train_and_evaluate_final(
    train_path="data/synthetic/train_smote_balanced.csv", 
    test_path="data/processed/test.csv",
    model_dir="models",
    report_dir="reports"
):
    print("🚀 Bắt đầu quy trình huấn luyện và kiểm thử cuối cùng...")
    
    # --- 1️⃣ Load dữ liệu ---
    # Load Train (đã SMOTE)
    if not os.path.exists(train_path):
        print(f"❌ Lỗi: Không tìm thấy file {train_path}")
        return
    df_train = pd.read_csv(train_path)
    
    # Load Test (Dữ liệu thực tế)
    if not os.path.exists(test_path):
        print(f"❌ Lỗi: Không tìm thấy file {test_path}")
        return
    df_test = pd.read_csv(test_path)

    print(f"📊 Dữ liệu Train (Synthetic): {len(df_train)} mẫu")
    print(f"📊 Dữ liệu Test (Real): {len(df_test)} mẫu")
    
    target_col = 'RF Link Quality'
    # Mapping để hiển thị cho đẹp
    class_names = ['Poor', 'Moderate', 'Good'] 
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    # --- 2️⃣ Huấn luyện Model ---
    print("\n🤖 Đang train model Random Forest...")
    rf_final = RandomForestClassifier(
        n_estimators=500, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced_subsample',
        max_depth=20,
        max_features='log2',
        min_samples_leaf=1,
        min_samples_split=2
    )
    rf_final.fit(X_train, y_train)
    
    # Lưu Model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_final, os.path.join(model_dir, "rf_final_model.pkl"))
    
    # --- 3️⃣ Đánh giá trên tập TEST (Quan trọng nhất) ---
    print("\n⚖️ Đang đánh giá trên tập Test thực tế...")
    y_pred = rf_final.predict(X_test)
    
    # Tính toán các chỉ số
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy trên tập Test: {acc:.2%}")
    
    print("\n📝 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # --- 4️⃣ Vẽ và Lưu Confusion Matrix ---
    os.makedirs(report_dir, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Dùng heatmap của seaborn cho đẹp hơn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title(f'Confusion Matrix (Test Set)\nAccuracy: {acc:.2%}')
    plt.ylabel('Thực tế (True Label)')
    plt.xlabel('Dự đoán (Predicted Label)')
    plt.tight_layout()
    
    cm_path = os.path.join(report_dir, "confusion_matrix_final.png")
    plt.savefig(cm_path)
    print(f"📊 Đã lưu Confusion Matrix tại: {cm_path}")
    
    # --- 5️⃣ Vẽ Feature Importance (Cập nhật lại) ---
    importances = rf_final.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df.head(10), x='Importance', y='Feature', palette='viridis')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    
    fi_path = os.path.join(report_dir, "feature_importance_final.png")
    plt.savefig(fi_path)
    print(f"📊 Đã lưu Feature Importance tại: {fi_path}")
    print("\n🎉 Hoàn tất quy trình!")

if __name__ == "__main__":
    train_and_evaluate_final()