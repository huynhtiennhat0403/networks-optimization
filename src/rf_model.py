import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
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
    if not os.path.exists(train_path):
        print(f"❌ Lỗi: Không tìm thấy file {train_path}")
        return
    df_train = pd.read_csv(train_path)
    
    if not os.path.exists(test_path):
        print(f"❌ Lỗi: Không tìm thấy file {test_path}")
        return
    df_test = pd.read_csv(test_path)

    print(f"📊 Dữ liệu Train (Synthetic): {len(df_train)} mẫu")
    print(f"📊 Dữ liệu Test (Real): {len(df_test)} mẫu")
    
    target_col = 'RF Link Quality'
    class_names = ['Poor', 'Moderate', 'Good']
    class_map = {'Poor': 0, 'Moderate': 1, 'Good': 2}
    
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
        min_samples_split=2,
        criterion='gini'
    )
    rf_final.fit(X_train, y_train)
    
    # Lưu Model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_final, os.path.join(model_dir, "rf_final_model.pkl"))
    
    # --- 3️⃣ Dự đoán ---
    print("\n⚖️ Đang đánh giá trên tập Test thực tế...")
    y_pred = rf_final.predict(X_test)
    y_pred_proba = rf_final.predict_proba(X_test)
    
    # --- 4️⃣ Tính toán các chỉ số chi tiết ---
    print("\n📊 TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ:")
    print("=" * 50)
    
    # 4.1 Accuracy tổng thể
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy (Độ chính xác tổng thể): {acc:.4f} ({acc:.2%})")
    
    # 4.2 Precision, Recall, F1 cho từng lớp
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print(f"\n📈 CHỈ SỐ THEO TỪNG LỚP:")
    print("-" * 60)
    print(f"{'Lớp':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        # Đếm số mẫu thực tế của mỗi lớp
        support = np.sum(y_test == i)
        print(f"{class_name:<10} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {support:<10}")
    
    # 4.3 Macro và Weighted Average
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 MACRO AVERAGE:")
    print(f"  Precision: {precision_macro:.4f} | Recall: {recall_macro:.4f} | F1-Score: {f1_macro:.4f}")
    
    print(f"📊 WEIGHTED AVERAGE:")
    print(f"  Precision: {precision_weighted:.4f} | Recall: {recall_weighted:.4f} | F1-Score: {f1_weighted:.4f}")
    
    # 4.4 AUC-ROC (cho multi-class)
    try:
        # Binarize labels cho AUC-ROC
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        auc_roc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
        print(f"\n📈 AUC-ROC Score (macro, One-vs-Rest): {auc_roc:.4f}")
    except Exception as e:
        print(f"\n⚠️ Không thể tính AUC-ROC: {e}")
    
    # 4.5 Classification Report đầy đủ
    print(f"\n📝 CLASSIFICATION REPORT ĐẦY ĐỦ:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # --- 5️⃣ Lưu kết quả vào file text ---
    os.makedirs(report_dir, exist_ok=True)
    
    report_content = f"""
=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST ===
Ngày đánh giá: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Mô hình: RandomForestClassifier (n_estimators=500, max_depth=20)
Dữ liệu huấn luyện: {train_path} ({len(df_train)} mẫu)
Dữ liệu kiểm thử: {test_path} ({len(df_test)} mẫu)

=== THỐNG KÊ DỮ LIỆU ===
Tập Train:
{y_train.value_counts().sort_index().to_string()}

Tập Test:
{y_test.value_counts().sort_index().to_string()}

=== CHỈ SỐ ĐÁNH GIÁ CHI TIẾT ===
1. Độ chính xác tổng thể (Accuracy): {acc:.4f} ({acc:.2%})

2. Chỉ số theo từng lớp:
{'Lớp':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}
{'-'*60}
"""
    
    for i, class_name in enumerate(class_names):
        support = np.sum(y_test == i)
        report_content += f"{class_name:<10} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} " \
                         f"{f1_per_class[i]:<12.4f} {support:<10}\n"
    
    report_content += f"""
3. Chỉ số tổng hợp:
- Macro Average:
  * Precision: {precision_macro:.4f}
  * Recall: {recall_macro:.4f}
  * F1-Score: {f1_macro:.4f}

- Weighted Average:
  * Precision: {precision_weighted:.4f}
  * Recall: {recall_weighted:.4f}
  * F1-Score: {f1_weighted:.4f}

4. AUC-ROC Score (macro, OvR): {auc_roc if 'auc_roc' in locals() else 'N/A'}

=== MA TRẬN NHẦM LẪN ===
{confusion_matrix(y_test, y_pred)}

=== THÔNG TIN MÔ HÌNH ===
- Số cây: 500
- Độ sâu tối đa: 20
- Số lượng features: {X_train.shape[1]}
- Đặc trưng quan trọng nhất: {X_train.columns[np.argmax(rf_final.feature_importances_)]}
- Overfit gap: {rf_final.score(X_train, y_train) - acc:.4f}

=== KẾT LUẬN ===
Mô hình đạt độ chính xác {acc:.2%} trên tập kiểm thử.
"""
    
    # Lưu file báo cáo
    report_path = os.path.join(report_dir, "model_performance_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\n📄 Đã lưu báo cáo chi tiết tại: {report_path}")
    
    # --- 6️⃣ Vẽ và lưu Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Số lượng mẫu'})
    
    plt.title(f'Confusion Matrix - Random Forest\nAccuracy: {acc:.2%}', fontsize=14, fontweight='bold')
    plt.ylabel('Nhãn thực tế', fontsize=12)
    plt.xlabel('Nhãn dự đoán', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(report_dir, "confusion_matrix_final.png")
    plt.savefig(cm_path, dpi=300)
    print(f"📊 Đã lưu Confusion Matrix tại: {cm_path}")
    
    # --- 7️⃣ Vẽ Feature Importance ---
    importances = rf_final.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(fi_df)), fi_df['Importance'], align='center', color='steelblue')
    plt.yticks(range(len(fi_df)), fi_df['Feature'])
    plt.xlabel('Mức độ quan trọng', fontsize=12)
    plt.title('Feature Importance trong Random Forest', fontsize=14, fontweight='bold')
    
    # Thêm giá trị số trên mỗi bar
    for i, (bar, importance) in enumerate(zip(bars, fi_df['Importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontsize=10)
    
    plt.gca().invert_yaxis()  # Đảo ngược để feature quan trọng nhất ở trên
    plt.tight_layout()
    
    fi_path = os.path.join(report_dir, "feature_importance_final.png")
    plt.savefig(fi_path, dpi=300)
    print(f"📊 Đã lưu Feature Importance tại: {fi_path}")
    
    # --- 8️⃣ Tạo bảng tổng hợp metrics ---
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'AUC-ROC'],
        'Value': [acc, precision_macro, recall_macro, f1_macro, auc_roc if 'auc_roc' in locals() else np.nan]
    })
    
    metrics_csv_path = os.path.join(report_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"📈 Đã lưu metrics vào CSV: {metrics_csv_path}")
    
    print(f"\n{'='*60}")
    print("🎉 HOÀN TẤT QUY TRÌNH ĐÁNH GIÁ!")
    print(f"{'='*60}")
    
    # Hiển thị thông tin tổng kết
    print(f"\n📋 TỔNG KẾT KẾT QUẢ:")
    print(f"  • Độ chính xác: {acc:.2%}")
    print(f"  • F1-Score (Macro): {f1_macro:.2%}")
    print(f"  • Recall lớp Good: {recall_per_class[2]:.2%}")
    print(f"  • Model đã lưu tại: models/rf_final_model.pkl")
    
    return rf_final, {
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

if __name__ == "__main__":
    model, metrics = train_and_evaluate_final()