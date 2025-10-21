import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ========= CONFIG =========
TRAIN_DATA_PATH = "data/synthetic/train_augmented.csv"
TEST_DATA_PATH = "data/processed/test.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "minmax_scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "onehot_encoder.pkl")

# Random Forest Hyperparameters
RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

SEED = 42
np.random.seed(SEED)

# Label mapping
LABEL_NAMES = {0: 'Poor', 1: 'Moderate', 2: 'Good'}


def load_data():
    """Load training and test data"""
    print("📂 Loading data...")
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"   Training samples: {len(train_df)}")
    
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"   Test samples: {len(test_df)}")
    
    return train_df, test_df


def prepare_features_and_labels(df, target_col="RF Link Quality"):
    """Separate features and target"""
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found!")
    
    # Features: all columns except target
    X = df.drop(columns=[target_col]).values
    
    # Target: RF Link Quality (already encoded as 0, 1, 2)
    y = df[target_col].astype(int).values
    
    # Get feature names for later use
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    return X, y, feature_names


def train_random_forest(X_train, y_train):
    """Train Random Forest Classifier"""
    print("\n🌲 Training Random Forest Classifier...")
    print(f"   Config: {RF_CONFIG}")
    
    clf = RandomForestClassifier(**RF_CONFIG)
    clf.fit(X_train, y_train)
    
    print("   ✅ Training completed!")
    return clf


def evaluate_model(clf, X_test, y_test):
    """Evaluate model performance"""
    print("\n📊 Evaluating model on test set...")
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"🎯 MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    
    # Classification report
    target_names = [LABEL_NAMES[i] for i in sorted(np.unique(y_test))]
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("🔢 Confusion Matrix:")
    print(cm)
    
    return accuracy, f1, cm, y_pred, y_pred_proba


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    labels = ['Poor', 'Moderate', 'Good']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - RF Link Quality Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_feature_importance(clf, feature_names, top_n=15, save_path=None):
    """Plot feature importance"""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Feature importance plot saved to: {save_path}")
    
    plt.close()
    
    # Print top features
    print("\n🔍 Top 10 Most Important Features:")
    for i, idx in enumerate(indices[:10], 1):
        print(f"   {i:2d}. {feature_names[idx]:<35} {importances[idx]:.4f}")


def save_model(clf, model_path):
    """Save trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"\n💾 Model saved to: {model_path}")


def test_prediction(clf, X_test, y_test, n_samples=5):
    """Test prediction on random samples"""
    print(f"\n🧪 Testing predictions on {n_samples} random samples:")
    print(f"{'='*60}")
    
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for i, idx in enumerate(indices, 1):
        X_sample = X_test[idx:idx+1]
        y_true = y_test[idx]
        y_pred = clf.predict(X_sample)[0]
        y_proba = clf.predict_proba(X_sample)[0]
        
        true_label = LABEL_NAMES[y_true]
        pred_label = LABEL_NAMES[y_pred]
        
        print(f"\nSample {i}:")
        print(f"   True:      {true_label}")
        print(f"   Predicted: {pred_label}")
        print(f"   Confidence: Poor={y_proba[0]:.2%}, Moderate={y_proba[1]:.2%}, Good={y_proba[2]:.2%}")
        print(f"   ✅ Correct" if y_true == y_pred else f"   ❌ Wrong")


def main():
    print("="*60)
    print("🚀 RANDOM FOREST MODEL TRAINING")
    print("="*60)
    
    # 1. Load data
    train_df, test_df = load_data()
    
    # Check class distribution
    print("\n📊 Training data class distribution:")
    class_dist = train_df["RF Link Quality"].value_counts().sort_index()
    for label, count in class_dist.items():
        print(f"   {LABEL_NAMES[int(label)]}: {count}")
    
    # 2. Prepare features and labels
    X_train, y_train, feature_names = prepare_features_and_labels(train_df)
    X_test, y_test, _ = prepare_features_and_labels(test_df)
    
    print(f"\n🔍 Dataset shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   Features: {len(feature_names)}")
    
    # 3. Train model
    clf = train_random_forest(X_train, y_train)
    
    # 4. Evaluate
    accuracy, f1, cm, y_pred, y_pred_proba = evaluate_model(clf, X_test, y_test)
    
    # 5. Visualizations
    plot_confusion_matrix(cm, save_path=os.path.join(REPORT_DIR, "confusion_matrix.png"))
    plot_feature_importance(clf, feature_names, save_path=os.path.join(REPORT_DIR, "feature_importance.png"))
    
    # 6. Save model
    save_model(clf, MODEL_PATH)
    
    # 7. Test predictions
    test_prediction(clf, X_test, y_test)
    
    # 8. Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Final Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"\n📦 Artifacts:")
    print(f"   Model:      {MODEL_PATH}")
    print(f"   Scaler:     {SCALER_PATH} (from preprocessing)")
    print(f"   Encoder:    {ENCODER_PATH} (from preprocessing)")
    print(f"\n🎯 Ready for deployment!")
    print("="*60)


if __name__ == "__main__":
    main()