"""
Comprehensive Evaluation Framework for Synthetic Data Quality
Đánh giá GAN qua 3 góc độ chính:
1. Statistical Similarity (Độ giống về mặt thống kê)
2. Machine Learning Utility (Hiệu quả khi train ML models)
3. Privacy/Diversity (Không bị duplicate real data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Label mapping
LABEL_NAMES = {0: 'Poor', 1: 'Moderate', 2: 'Good'}

# ========================================
# 1. STATISTICAL SIMILARITY TESTS
# ========================================

def evaluate_statistical_similarity(real_df, syn_df, features, target_col):
    """
    Đánh giá độ giống nhau về mặt thống kê giữa real và synthetic data
    """
    print("\n" + "="*80)
    print("📊 1. STATISTICAL SIMILARITY EVALUATION")
    print("="*80)
    
    results = {
        'feature': [],
        'ks_statistic': [],
        'ks_pvalue': [],
        'wasserstein_distance': [],
        'mean_diff': [],
        'std_diff': []
    }
    
    for feat in features:
        real_vals = real_df[feat].values
        syn_vals = syn_df[feat].values
        
        # Kolmogorov-Smirnov test (kiểm tra phân phối có giống nhau không)
        ks_stat, ks_p = stats.ks_2samp(real_vals, syn_vals)
        
        # Wasserstein distance (khoảng cách giữa 2 phân phối)
        w_dist = stats.wasserstein_distance(real_vals, syn_vals)
        
        # Basic statistics
        mean_diff = abs(real_vals.mean() - syn_vals.mean())
        std_diff = abs(real_vals.std() - syn_vals.std())
        
        results['feature'].append(feat)
        results['ks_statistic'].append(ks_stat)
        results['ks_pvalue'].append(ks_p)
        results['wasserstein_distance'].append(w_dist)
        results['mean_diff'].append(mean_diff)
        results['std_diff'].append(std_diff)
    
    results_df = pd.DataFrame(results)
    
    # Summary
    print("\n📈 Top 5 features with BEST similarity (lowest Wasserstein distance):")
    print(results_df.nsmallest(5, 'wasserstein_distance')[['feature', 'wasserstein_distance', 'ks_pvalue']])
    
    print("\n⚠️  Top 5 features with WORST similarity (highest Wasserstein distance):")
    print(results_df.nlargest(5, 'wasserstein_distance')[['feature', 'wasserstein_distance', 'ks_pvalue']])
    
    # Overall score
    avg_w_dist = results_df['wasserstein_distance'].mean()
    pct_similar = (results_df['ks_pvalue'] > 0.05).sum() / len(results_df) * 100
    
    print(f"\n📊 OVERALL STATISTICAL SIMILARITY:")
    print(f"  • Average Wasserstein Distance: {avg_w_dist:.4f}")
    print(f"  • % Features with similar distribution (p>0.05): {pct_similar:.1f}%")
    
    # Distribution comparison for target variable
    print(f"\n🎯 Target Distribution Comparison ({target_col}):")
    real_dist = real_df[target_col].value_counts(normalize=True).sort_index()
    syn_dist = syn_df[target_col].value_counts(normalize=True).sort_index()
    
    comp_df = pd.DataFrame({
        'Real': real_dist,
        'Synthetic': syn_dist,
        'Diff': abs(real_dist - syn_dist)
    })
    
    # Add label names to index
    comp_df.index = [LABEL_NAMES[int(idx)] for idx in comp_df.index]
    print(comp_df)
    
    # Jensen-Shannon divergence (0=identical, 1=completely different)
    js_div = jensenshannon(real_dist.values, syn_dist.values)
    print(f"\n  Jensen-Shannon Divergence: {js_div:.4f} (lower is better, <0.1 is excellent)")
    
    return results_df, js_div


# ========================================
# 2. MACHINE LEARNING UTILITY TEST (TRST)
# ========================================

def evaluate_ml_utility(real_df, syn_df, test_df, features, target_col):
    """
    Train on Real, Test on Synthetic (TRTS)
    Train on Synthetic, Test on Real (TSTR)
    """
    print("\n" + "="*80)
    print("🤖 2. MACHINE LEARNING UTILITY EVALUATION")
    print("="*80)
    
    # Prepare data - Target đã là 0, 1, 2 rồi, không cần map
    X_real = real_df[features].values
    y_real = real_df[target_col].astype(int).values
    
    X_syn = syn_df[features].values
    y_syn = syn_df[target_col].astype(int).values
    
    X_test = test_df[features].values
    y_test = test_df[target_col].astype(int).values
    
    # Multiple classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=10),
    }
    
    results = []
    
    for clf_name, clf in classifiers.items():
        print(f"\n--- {clf_name} ---")
        
        # Baseline: Train on Real only, Test on Real test set
        clf_real = clf.__class__(**clf.get_params())
        clf_real.fit(X_real, y_real)
        y_pred_real = clf_real.predict(X_test)
        acc_real = accuracy_score(y_test, y_pred_real)
        f1_real = f1_score(y_test, y_pred_real, average='weighted')
        
        # TSTR: Train on Synthetic, Test on Real
        clf_syn = clf.__class__(**clf.get_params())
        clf_syn.fit(X_syn, y_syn)
        y_pred_tstr = clf_syn.predict(X_test)
        acc_tstr = accuracy_score(y_test, y_pred_tstr)
        f1_tstr = f1_score(y_test, y_pred_tstr, average='weighted')
        
        # Combined: Train on Real + Synthetic, Test on Real
        X_combined = np.vstack([X_real, X_syn])
        y_combined = np.concatenate([y_real, y_syn])
        clf_combined = clf.__class__(**clf.get_params())
        clf_combined.fit(X_combined, y_combined)
        y_pred_combined = clf_combined.predict(X_test)
        acc_combined = accuracy_score(y_test, y_pred_combined)
        f1_combined = f1_score(y_test, y_pred_combined, average='weighted')
        
        print(f"  Real only:        Acc={acc_real:.4f}, F1={f1_real:.4f}")
        print(f"  Synthetic only:   Acc={acc_tstr:.4f}, F1={f1_tstr:.4f} (utility={acc_tstr/acc_real:.2%})")
        print(f"  Real+Synthetic:   Acc={acc_combined:.4f}, F1={f1_combined:.4f} (gain={acc_combined-acc_real:+.4f})")
        
        results.append({
            'Classifier': clf_name,
            'Real_Acc': acc_real,
            'Synthetic_Acc': acc_tstr,
            'Combined_Acc': acc_combined,
            'Real_F1': f1_real,
            'Synthetic_F1': f1_tstr,
            'Combined_F1': f1_combined,
            'Utility_Ratio': acc_tstr / acc_real,
            'Improvement': acc_combined - acc_real
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n📊 SUMMARY:")
    print(f"  Average Utility Ratio (Synthetic/Real): {results_df['Utility_Ratio'].mean():.2%}")
    print(f"  Average Improvement (Combined vs Real): {results_df['Improvement'].mean():+.4f}")
    
    if results_df['Utility_Ratio'].mean() > 0.85:
        print("  ✅ EXCELLENT: Synthetic data has high utility!")
    elif results_df['Utility_Ratio'].mean() > 0.70:
        print("  ⚠️  GOOD: Synthetic data is usable but could be better")
    else:
        print("  ❌ POOR: Synthetic data quality needs improvement")
    
    return results_df


# ========================================
# 3. PRIVACY & DIVERSITY CHECK
# ========================================

def evaluate_privacy_diversity(real_df, syn_df, features, n_samples=1000):
    """
    Kiểm tra xem synthetic data có bị duplicate real data không
    """
    print("\n" + "="*80)
    print("🔒 3. PRIVACY & DIVERSITY EVALUATION")
    print("="*80)
    
    # Sample để tính toán nhanh hơn
    real_sample = real_df[features].sample(min(n_samples, len(real_df)), random_state=42).values
    syn_sample = syn_df[features].sample(min(n_samples, len(syn_df)), random_state=42).values
    
    # Calculate minimum distances
    from sklearn.metrics.pairwise import euclidean_distances
    
    print("\n⏳ Computing nearest neighbor distances...")
    distances = euclidean_distances(syn_sample, real_sample)
    min_distances = distances.min(axis=1)
    
    # Statistics
    print(f"\n📊 Distance Statistics (synthetic to nearest real sample):")
    print(f"  • Mean distance:   {min_distances.mean():.4f}")
    print(f"  • Median distance: {np.median(min_distances):.4f}")
    print(f"  • Min distance:    {min_distances.min():.4f}")
    print(f"  • Max distance:    {min_distances.max():.4f}")
    
    # Check for potential duplicates (very close samples)
    threshold = 0.01
    n_close = (min_distances < threshold).sum()
    pct_close = n_close / len(min_distances) * 100
    
    print(f"\n🔍 Samples very close to real data (distance < {threshold}):")
    print(f"  • Count: {n_close} / {len(min_distances)} ({pct_close:.2f}%)")
    
    if pct_close < 1:
        print("  ✅ EXCELLENT: Very low risk of memorization!")
    elif pct_close < 5:
        print("  ⚠️  GOOD: Acceptable diversity")
    else:
        print("  ❌ WARNING: High risk of memorization/overfitting!")
    
    # Diversity within synthetic data
    syn_distances = euclidean_distances(syn_sample[:500], syn_sample[:500])
    np.fill_diagonal(syn_distances, np.inf)  # Ignore self-distances
    syn_min_dist = syn_distances.min(axis=1).mean()
    
    print(f"\n🎨 Intra-synthetic diversity:")
    print(f"  • Average nearest neighbor distance: {syn_min_dist:.4f}")
    
    return min_distances


# ========================================
# 4. VISUALIZATION
# ========================================

def visualize_comparison(real_df, syn_df, features, target_col):
    """
    Visualize real vs synthetic data
    """
    print("\n" + "="*80)
    print("📈 4. GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Prepare data for PCA
    X_real = real_df[features].values
    X_syn = syn_df[features].values
    
    # PCA projection
    print("\n⏳ Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.vstack([X_real, X_syn]))
    
    X_real_pca = pca.transform(X_real)
    X_syn_pca = pca.transform(X_syn)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. PCA scatter plot
    ax = axes[0, 0]
    ax.scatter(X_real_pca[:, 0], X_real_pca[:, 1], alpha=0.5, label='Real', s=20)
    ax.scatter(X_syn_pca[:, 0], X_syn_pca[:, 1], alpha=0.5, label='Synthetic', s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA Projection: Real vs Synthetic')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Distribution comparison for top features
    ax = axes[0, 1]
    top_features = features[:3]  # First 3 continuous features
    positions = np.arange(len(top_features))
    width = 0.35
    
    real_means = [real_df[f].mean() for f in top_features]
    syn_means = [syn_df[f].mean() for f in top_features]
    
    ax.bar(positions - width/2, real_means, width, label='Real', alpha=0.7)
    ax.bar(positions + width/2, syn_means, width, label='Synthetic', alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f[:15] for f in top_features], rotation=45, ha='right')
    ax.set_ylabel('Mean Value')
    ax.set_title('Feature Means Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Target distribution
    ax = axes[1, 0]
    real_dist = real_df[target_col].value_counts().sort_index()
    syn_dist = syn_df[target_col].value_counts().sort_index()
    
    x = np.arange(len(real_dist))
    ax.bar(x - width/2, real_dist.values, width, label='Real', alpha=0.7)
    ax.bar(x + width/2, syn_dist.values, width, label='Synthetic', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_NAMES[int(i)] for i in real_dist.index])
    ax.set_ylabel('Count')
    ax.set_title(f'{target_col} Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Box plot comparison for key features
    ax = axes[1, 1]
    feature_to_compare = features[0]  # First continuous feature
    data_to_plot = [
        real_df[feature_to_compare].values,
        syn_df[feature_to_compare].values
    ]
    ax.boxplot(data_to_plot, labels=['Real', 'Synthetic'])
    ax.set_ylabel('Value')
    ax.set_title(f'{feature_to_compare[:30]} Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/gan_evaluation_plots.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plots saved to: reports/gan_evaluation_plots.png")
    plt.show()


# ========================================
# MAIN EVALUATION PIPELINE
# ========================================

def run_full_evaluation(real_path, synthetic_path, test_path):
    """
    Run complete evaluation pipeline
    """
    print("\n" + "="*80)
    print("🚀 STARTING COMPREHENSIVE GAN EVALUATION")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    real_df = pd.read_csv(real_path)
    syn_df = pd.read_csv(synthetic_path)
    test_df = pd.read_csv(test_path)
    
    print(f"  Real data: {len(real_df)} samples")
    print(f"  Synthetic data: {len(syn_df)} samples")
    print(f"  Test data: {len(test_df)} samples")
    
    # Identify features
    target_col = "RF Link Quality"
    features = [c for c in real_df.columns if c != target_col]
    
    print(f"\n  Features: {len(features)}")
    print(f"  Target: {target_col}")
    
    # Check target values
    print(f"\n  Real target distribution:")
    for val, count in real_df[target_col].value_counts().sort_index().items():
        print(f"    {LABEL_NAMES[int(val)]} ({val}): {count}")
    
    # Run all evaluations
    results = {}
    
    # 1. Statistical Similarity
    stat_results, js_div = evaluate_statistical_similarity(real_df, syn_df, features, target_col)
    results['statistical'] = {'js_divergence': js_div}
    
    # 2. ML Utility
    ml_results = evaluate_ml_utility(real_df, syn_df, test_df, features, target_col)
    results['ml_utility'] = ml_results
    
    # 3. Privacy & Diversity
    distances = evaluate_privacy_diversity(real_df, syn_df, features)
    results['privacy'] = {'mean_distance': distances.mean()}
    
    # 4. Visualization
    visualize_comparison(real_df, syn_df, features, target_col)
    
    # Final Summary
    print("\n" + "="*80)
    print("🏆 FINAL EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n1️⃣  Statistical Similarity:")
    print(f"    JS Divergence: {js_div:.4f} {'✅ Excellent' if js_div < 0.1 else '⚠️  Needs improvement'}")
    
    print(f"\n2️⃣  ML Utility:")
    avg_utility = ml_results['Utility_Ratio'].mean()
    avg_improvement = ml_results['Improvement'].mean()
    print(f"    Utility Ratio: {avg_utility:.2%} {'✅ Good' if avg_utility > 0.85 else '⚠️  Fair'}")
    print(f"    Avg Improvement: {avg_improvement:+.4f} {'✅ Helpful!' if avg_improvement > 0 else '⚠️  No gain'}")
    
    print(f"\n3️⃣  Privacy & Diversity:")
    print(f"    Mean distance to real: {distances.mean():.4f} ✅")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    
    return results


# ========================================
# USAGE
# ========================================

if __name__ == "__main__":
    # Paths to your data
    REAL_PATH = "data/processed/train.csv"
    SYNTHETIC_PATH = "data/synthetic/train_augmented.csv"
    TEST_PATH = "data/processed/test.csv"
    
    # Run evaluation
    results = run_full_evaluation(REAL_PATH, SYNTHETIC_PATH, TEST_PATH)