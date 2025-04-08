import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
from sklearn.preprocessing import StandardScaler

# 1. 数据加载与预处理
data = pd.read_csv('附件1：ReplicatedAcousticFeatures-ParkinsonDatabase.csv')
labels = data['Status']
subject_mean = data.groupby(['ID', 'Status', 'Gender']).mean().reset_index()


# 2. 特征分析函数
def analyze_features(data):
    feature_groups = {
        'Jitter': ['Jitter_rel', 'Jitter_abs', 'Jitter_RAP', 'Jitter_PPQ'],
        'Shimmer': ['Shim_loc', 'Shim_dB', 'Shim_APQ3', 'Shim_APQ5', 'Shim_APQ11'],
        'HNR': ['HNR05', 'HNR15', 'HNR25', 'HNR35', 'HNR38'],
        'Nonlinear': ['RPDE', 'DFA', 'PPE', 'GNE'],
        'MFCC': [f'MFCC{i}' for i in range(13)],
        'Delta': [f'Delta{i}' for i in range(13)]
    }

    results = []
    for group, features in feature_groups.items():
        for feat in features:
            healthy = data[data['Status'] == 0][feat]
            pd_patients = data[data['Status'] == 1][feat]

            _, p_normal_healthy = stats.shapiro(healthy)
            _, p_normal_pd = stats.shapiro(pd_patients)

            if p_normal_healthy > 0.05 and p_normal_pd > 0.05:
                _, p_val = stats.ttest_ind(healthy, pd_patients)
                test_type = 't-test'
            else:
                _, p_val = stats.mannwhitneyu(healthy, pd_patients)
                test_type = 'Mann-Whitney'

            cohen_d = (pd_patients.mean() - healthy.mean()) / np.sqrt((healthy.std() ** 2 + pd_patients.std() ** 2) / 2)
            results.append({
                'Feature': feat, 'Group': group,
                'Healthy_mean': healthy.mean(), 'PD_mean': pd_patients.mean(),
                'p_value': p_val, 'Effect_size': cohen_d, 'Test': test_type
            })

    return pd.DataFrame(results), feature_groups


# 3. 执行分析并可视化
results_df, feature_groups = analyze_features(subject_mean)
significant_features = results_df[(results_df['p_value'] < 0.01) & (abs(results_df['Effect_size']) > 0.5)].sort_values(
    'p_value')

# 保存关键特征箱线图
plt.figure(figsize=(30, 20))
for i, feat in enumerate(significant_features['Feature'][:6], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Status', y=feat, data=data)
    plt.title(f'{feat} (p={significant_features.iloc[i - 1]["p_value"]:.3e})')
    plt.xticks([0, 1], ['Healthy', 'PD'])
plt.tight_layout()
plt.savefig('significant_features_boxplot.svg', format="svg", dpi=300)
plt.close()

# 保存特征相关性热图
plt.figure(figsize=(24, 20))
corr_matrix = data[significant_features['Feature']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig('feature_correlation_heatmap.svg', format='svg', dpi=300)
plt.close()

# ROC曲线评估(简化版)
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))
for feat in significant_features['Feature'][:4]:
    fpr, tpr, _ = roc_curve(labels, data[feat])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{feat} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('关键特征ROC曲线')
plt.legend(loc='lower right')
plt.show()

# 保存结果
with open('feature_analysis_results.pkl', 'wb') as f:
    pickle.dump({
        'results_df': results_df,
        'significant_features': significant_features,
        'feature_groups': feature_groups,
        'subject_mean': subject_mean,
        'data': data
    }, f)

