# ======================
# 第二问：诊断模型构建
# ======================

import pymc3 as pm
import arviz as az
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# 1. 加载第一问结果
with open('feature_analysis_results.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    significant_features = saved_data['significant_features']
    feature_groups = saved_data['feature_groups']
    subject_mean = saved_data['subject_mean']
    data = saved_data['data']

# 2. 特征选择
selected_features = []
for group, features in feature_groups.items():
    group_features = significant_features[significant_features['Group'] == group]
    if not group_features.empty:
        best_feature = group_features.iloc[0]['Feature']
        selected_features.append(best_feature)

# 3. 准备数据
X = subject_mean[selected_features].values
y = subject_mean['Status'].values
gender = subject_mean['Gender'].values
subject_idx = pd.factorize(subject_mean['ID'])[0]

# 4. 贝叶斯模型构建
with pm.Model() as model:
    mu_w = pm.Normal('mu_w', mu=0, sigma=10, shape=len(selected_features))
    sigma_w = pm.HalfNormal('sigma_w', sigma=1, shape=len(selected_features))
    w = pm.Normal('w', mu=mu_w, sigma=sigma_w, shape=(len(subject_mean), len(selected_features)))
    beta_x = pm.Laplace('beta_x', mu=0, b=1, shape=len(selected_features))
    beta_z = pm.Normal('beta_z', mu=0, sigma=10)
    mu = pm.math.dot(w, beta_x) + beta_z * gender
    p = pm.math.invprobit(mu)
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
    trace = pm.sample(2000, tune=1000, chains=2, target_accept=0.9)

# 5. 模型评估与可视化
# 轨迹图
az.plot_trace(trace, var_names=['beta_x', 'beta_z'])
plt.tight_layout()
plt.savefig('trace_plot.png')
plt.close()

# 特征重要性
beta_x_post = trace['beta_x'].mean(axis=0)
sorted_idx = np.argsort(-np.abs(beta_x_post))
plt.figure(figsize=(10, 6))
plt.barh(np.array(selected_features)[sorted_idx], beta_x_post[sorted_idx])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
for train_idx, test_idx in kfold.split(X, y):
    with model:
        pm.set_data({'w': X[train_idx]})
        approx = pm.fit(n=3000, method='advi')
        ppc = approx.sample_posterior_predictive()
        y_pred = ppc['y_obs'].mean(axis=0) > 0.5
        cv_results.append({
            'accuracy': accuracy_score(y[test_idx], y_pred),
            'auc': roc_auc_score(y[test_idx], ppc['y_obs'].mean(axis=0))
        })

# 最终评估
y_pred_proba = pm.math.invprobit(np.dot(X, beta_x_post) + trace['beta_z'].mean() * gender).eval()
y_pred = (y_pred_proba > 0.5).astype(int)

# 混淆矩阵
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Healthy', 'Predicted PD'],
            yticklabels=['Actual Healthy', 'Actual PD'])
plt.title(f'Confusion Matrix (Accuracy={accuracy_score(y, y_pred):.2f})')
plt.savefig('confusion_matrix.png')
plt.close()

# ROC曲线
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# 保存模型结果
with open('model_results.pkl', 'wb') as f:
    pickle.dump({
        'selected_features': selected_features,
        'trace': trace,
        'cv_results': cv_results,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }, f)