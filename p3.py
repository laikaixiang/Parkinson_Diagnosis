import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. 加载前两问结果
with open('feature_analysis_results.pkl', 'rb') as f:
    feature_data = pickle.load(f)
    significant_features = feature_data['significant_features']
    data = feature_data['data']

with open('model_results.pkl', 'rb') as f:
    model_data = pickle.load(f)
    selected_features = model_data['selected_features']

# 2. 准备聚类数据（仅PD患者）
pd_patients = data[data['Status'] == 1].copy()
X = pd_patients[selected_features].values

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 两类聚类（运动型 vs 非运动型）
kmeans_2 = KMeans(n_clusters=2, random_state=42)
cluster_2_labels = kmeans_2.fit_predict(X_scaled)
pd_patients['Main_Cluster'] = cluster_2_labels

# 定义聚类类型（基于特征分析）
cluster_types = {
    0: 'Motor-Dominant',
    1: 'Non-Motor-Dominant'
}
pd_patients['Main_Cluster_Type'] = pd_patients['Main_Cluster'].map(cluster_types)

# 4. 三类子聚类
final_clusters = []
for main_cluster in [0, 1]:
    # 提取当前主类的数据
    mask = pd_patients['Main_Cluster'] == main_cluster
    X_sub = X_scaled[mask]

    # 子聚类
    kmeans_3 = KMeans(n_clusters=3, random_state=42)
    sub_labels = kmeans_3.fit_predict(X_sub)

    # 转换为全局标签 (0-5)
    global_labels = sub_labels + (main_cluster * 3)
    final_clusters.extend(global_labels)

pd_patients['Sub_Cluster'] = final_clusters

# 5. 定义症状子类型
symptom_subtypes = {
    0: 'Tremor-Dominant',
    1: 'Bradykinesia-Dominant',
    2: 'Rigidity-Dominant',
    3: 'Pain-Dominant',
    4: 'Dementia-Dominant',
    5: 'Sleep-Disorder'
}
pd_patients['Symptom_Subtype'] = pd_patients['Sub_Cluster'].map(symptom_subtypes)

# 6. 可视化聚类结果
# PCA降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 6))

# 主类可视化
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=pd_patients['Main_Cluster_Type'],
                palette='Set1', s=100)
plt.title('Main Clusters (Motor vs Non-Motor)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# 子类可视化
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=pd_patients['Symptom_Subtype'],
                palette='Dark2', s=100)
plt.title('Symptom Subtypes')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.savefig('clustering_results.png')
plt.close()

# 7. 保存结果
# 保存聚类结果到CSV
cluster_results = pd_patients[['ID', 'Recording', 'Main_Cluster', 'Main_Cluster_Type',
                               'Sub_Cluster', 'Symptom_Subtype'] + selected_features]
cluster_results.to_csv('symptom_clustering_results.csv', index=False)

# 保存中间数据到PKL
with open('clustering_data.pkl', 'wb') as f:
    pickle.dump({
        'X_scaled': X_scaled,
        'cluster_2_labels': cluster_2_labels,
        'final_clusters': final_clusters,
        'pca_components': X_pca,
        'selected_features': selected_features
    }, f)

# 8. 各类别特征分析（保存到CSV）
cluster_stats = []
for subtype in symptom_subtypes.values():
    subtype_data = pd_patients[pd_patients['Symptom_Subtype'] == subtype]
    stats = subtype_data[selected_features].mean().to_dict()
    stats['Subtype'] = subtype
    stats['Count'] = len(subtype_data)
    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)
cluster_stats_df.to_csv('subtype_feature_stats.csv', index=False)