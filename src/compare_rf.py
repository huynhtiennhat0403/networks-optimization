import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import os

def train_and_evaluate(
    train_original_path="data/processed/train.csv",
    train_synthetic_path="data/synthetic/train_smote_balanced.csv",
    test_path="data/processed/test.csv",
    output_dir="reports"
):
    print("🚀 Bắt đầu quy trình kiểm thử 3 kịch bản...")
    
    # --- 1️⃣ Chuẩn bị dữ liệu ---
    # Load dữ liệu
    if not os.path.exists(train_original_path) or not os.path.exists(train_synthetic_path) or not os.path.exists(test_path):
        print("❌ Lỗi: Không tìm thấy file dữ liệu. Hãy đảm bảo bạn đã chạy processing_data.py và smote.py")
        return

    df_train_orig = pd.read_csv(train_original_path)
    df_train_syn = pd.read_csv(train_synthetic_path)
    df_test = pd.read_csv(test_path)
    
    # Kịch bản 3: Combine (Gốc + Synthetic)
    df_train_combined = pd.concat([df_train_orig, df_train_syn], axis=0).reset_index(drop=True)
    
    # Dictionary chứa 3 bộ dữ liệu train
    datasets = {
        "1. Original Data": df_train_orig,
        "2. Synthetic Only": df_train_syn,
        "3. Combined (SMOTE)": df_train_combined
    }
    
    target_col = 'RF Link Quality'
    class_names = ['Poor', 'Moderate', 'Good']
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    # --- 2️⃣ Train & Đánh giá ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    results_txt = []
    
    print(f"\n{'='*20} TRAINING {'='*20}")
    
    for i, (name, df_train) in enumerate(datasets.items()):
        print(f"🤖 Đang train Model: {name} (Số mẫu: {len(df_train)})")
        
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        
        # Train
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Tính Accuracy NGAY TẠI ĐÂY để hiển thị
        acc = accuracy_score(y_test, y_pred)
        
        # Vẽ Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        
        # Màu sắc khác nhau cho mỗi biểu đồ
        cmap = ['Blues', 'Oranges', 'Greens'][i]
        disp.plot(ax=axes[i], cmap=cmap, values_format='d')
        
        # --- CẬP NHẬT TIÊU ĐỀ CÓ ACCURACY ---
        axes[i].set_title(f"{name}\n({len(df_train)} samples)\nAccuracy: {acc:.2%}", fontsize=12, fontweight='bold')
        
        # Lưu kết quả text để in ra sau
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        res_str = f"\n🔹 MODEL: {name}\n"
        res_str += f"   - Accuracy: {acc:.2%}\n"
        res_str += f"   - Recall (Good): {report['Good']['recall']:.2%}\n"
        res_str += f"   - Recall (Moderate): {report['Moderate']['recall']:.2%}\n"
        res_str += f"   - F1-Score (Good): {report['Good']['f1-score']:.2f}"
        results_txt.append(res_str)

    # --- 3️⃣ Lưu biểu đồ ---
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'comparison_3_models.png')
    plt.savefig(plot_path)
    print(f"\n📊 Đã lưu biểu đồ so sánh tại: {plot_path}")
    
    # --- 4️⃣ In kết quả chi tiết ---
    print(f"\n{'='*20} KẾT QUẢ CHI TIẾT {'='*20}")
    for res in results_txt:
        print(res)
    print("="*60)

if __name__ == "__main__":
    train_and_evaluate()