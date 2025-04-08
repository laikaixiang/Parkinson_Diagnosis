import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. 加载预处理数据和模型
with open('feature_analysis_results.pkl', 'rb') as f:
    feature_data = pickle.load(f)
    data = feature_data['data']
    selected_features = feature_data['significant_features']['Feature'].tolist()

# 2. 准备诊断数据
X = data[selected_features].values
y_true = data['Status'].values
subject_ids = data['ID'].values
recordings = data['Recording'].values
gender = data['Gender'].values.reshape(-1, 1)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 添加性别作为特征
X_final = np.hstack([X_scaled, gender])

# 3. 加载训练好的模型
with open('model_results.pkl', 'rb') as f:
    model_data = pickle.load(f)
    best_model_name = max(model_data['results'].items(), key=lambda x: x[1]['auc'])[0]

    # 重建最佳模型
    if best_model_name == "Logistic Regression":
        model = LogisticRegression(**model_data['models'][best_model_name])
    elif best_model_name == "Random Forest":
        model = RandomForestClassifier(**model_data['models'][best_model_name])
    else:  # XGBoost
        model = XGBClassifier(**model_data['models'][best_model_name])

    # 使用完整数据重新训练模型（因为p2.py中使用的是交叉验证）
    model.fit(X_final, y_true)

# 4. 进行诊断预测
pred_proba = model.predict_proba(X_final)[:, 1]
y_pred = (pred_proba > 0.5).astype(int)

# 5. 评估诊断性能
performance = {
    'accuracy': accuracy_score(y_true, y_pred),
    'auc': roc_auc_score(y_true, pred_proba),
    'confusion_matrix': confusion_matrix(y_true, y_pred),
    'classification_report': classification_report(y_true, y_pred, output_dict=True),
    'model_used': best_model_name
}

# 6. 结果可视化
# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = performance['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Healthy', 'Predicted PD'],
            yticklabels=['Actual Healthy', 'Actual PD'])
plt.title(f'Confusion Matrix ({best_model_name})\nAccuracy={performance["accuracy"]:.2f}')
plt.savefig('diagnosis_confusion_matrix.png')
plt.close()

# ROC曲线
fpr, tpr, _ = roc_curve(y_true, pred_proba)
roc_auc = performance['auc']
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic ({best_model_name})')
plt.legend(loc='lower right')
plt.savefig('diagnosis_roc_curve.png')
plt.close()

# 7. 保存诊断结果
diagnosis_results = pd.DataFrame({
    'ID': subject_ids,
    'Recording': recordings,
    'True_Status': y_true,
    'Predicted_Probability': pred_proba,
    'Predicted_Status': y_pred
})

# 添加特征值
for feat in selected_features:
    diagnosis_results[feat] = data[feat].values

diagnosis_results.to_csv('full_diagnosis_results.csv', index=False)

# 8. 保存性能指标
with open('diagnosis_performance.pkl', 'wb') as f:
    pickle.dump(performance, f)

# 9. 按受试者聚合结果(每人3次录音)
subject_level_results = diagnosis_results.groupby('ID').agg({
    'True_Status': 'first',
    'Predicted_Probability': 'mean',
    'Predicted_Status': lambda x: (x.mean() > 0.5).astype(int)
}).reset_index()

subject_level_results.to_csv('subject_level_diagnosis.csv', index=False)

# 10. 打印关键性能指标
print("\n=== 诊断模型性能 ===")
print(f"使用的模型: {best_model_name}")
print(f"准确率: {performance['accuracy']:.3f}")
print(f"AUC: {performance['auc']:.3f}")
print("\n分类报告:")
print(pd.DataFrame(performance['classification_report']).transpose())