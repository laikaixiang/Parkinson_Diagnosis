import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, roc_curve, auc)
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# 1. 加载数据
with open('feature_analysis_results.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    significant_features = saved_data['significant_features']
    feature_groups = saved_data['feature_groups']
    subject_mean = saved_data['subject_mean']

# 2. 特征选择
selected_features = []
for group, features in feature_groups.items():
    group_features = significant_features[significant_features['Group'] == group]
    if not group_features.empty:
        best_feature = group_features.iloc[0]['Feature']
        selected_features.append(best_feature)

print("最终选择的特征:", selected_features)

# 3. 准备数据
X = subject_mean[selected_features].values
y = subject_mean['Status'].values
gender = subject_mean['Gender'].values.reshape(-1, 1)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 添加性别作为特征
X_final = np.hstack([X_scaled, gender])

# 4. 定义三种模型
models = {
    "Logistic Regression": LogisticRegression(penalty='l2', C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, random_state=42, use_label_encoder=False)
}

# 5. 交叉验证评估
results = {}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n正在评估 {name}...")

    # 获取预测概率（使用交叉验证）
    y_proba = cross_val_predict(model, X_final, y, cv=kfold, method='predict_proba')[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    # 计算指标
    acc = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    # 存储结果
    results[name] = {
        'accuracy': acc,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'y_proba': y_proba
    }

    # 打印结果
    print(f"{name} 结果:")
    print(f"准确率: {acc:.3f}")
    print(f"AUC: {roc_auc:.3f}")
    print("混淆矩阵:")
    print(cm)


# 6. 可视化结果
def plot_results(results):
    # 性能比较
    metrics = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [x['accuracy'] for x in results.values()],
        'AUC': [x['auc'] for x in results.values()]
    }).melt(id_vars='Model', var_name='Metric')

    plt.figure(figsize=(12, 5))
    sns.barplot(x='Model', y='value', hue='Metric', data=metrics)
    plt.title('模型性能比较')
    plt.ylabel('分数')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    # ROC曲线
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res['y_proba'])
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {res["auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线比较')
    plt.legend(loc='lower right')
    plt.savefig('roc_comparison.png')
    plt.close()

    # 混淆矩阵
    plt.figure(figsize=(15, 4))
    for i, (name, res) in enumerate(results.items(), 1):
        plt.subplot(1, 3, i)
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Healthy', 'Predicted PD'],
                    yticklabels=['Actual Healthy', 'Actual PD'])
        plt.title(f'{name}\nAccuracy: {res["accuracy"]:.2f}')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()


plot_results(results)

# 7. 保存结果
with open('model_results.pkl', 'wb') as f:
    pickle.dump({
        'selected_features': selected_features,
        'results': results,
        'models': {k: v.get_params() for k, v in models.items()}
    }, f)

print("\n分析完成！结果已保存。")