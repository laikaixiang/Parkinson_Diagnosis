import pandas as pd
import numpy as np
import pickle
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

# 1. 加载必要数据
with open('feature_analysis_results.pkl', 'rb') as f:
    feature_data = pickle.load(f)
    data = feature_data['data']
    selected_features = feature_data['significant_features']['Feature'].tolist()
    feature_groups = feature_data['feature_groups']

# 加载第三问的聚类结果
cluster_results = pd.read_csv('symptom_clustering_results.csv')
data = pd.merge(data, cluster_results[['ID', 'Recording', 'Symptom_Subtype']],
                on=['ID', 'Recording'], how='left')

# 2. 仅分析PD患者数据
pd_data = data[data['Status'] == 1].copy()

# 3. 定义症状类型（根据聚类结果）
symptom_subtypes = pd_data['Symptom_Subtype'].unique().tolist()
motor_subtypes = [st for st in symptom_subtypes if
                  st in ['Tremor-Dominant', 'Bradykinesia-Dominant', 'Rigidity-Dominant']]
non_motor_subtypes = [st for st in symptom_subtypes if st in ['Pain-Dominant', 'Dementia-Dominant', 'Sleep-Disorder']]


# 4. 特征统计分析函数
def analyze_symptom_features(pd_data, feature_groups):
    results = []

    for group_name, features in feature_groups.items():
        for feature in features:
            temp_df = pd_data[['Symptom_Subtype', feature]].dropna()
            if temp_df['Symptom_Subtype'].nunique() < 2:
                continue

            # ANOVA检验
            model = ols(f'{feature} ~ C(Symptom_Subtype)', data=temp_df).fit()
            anova_p = sm.stats.anova_lm(model, typ=2)['PR(>F)'][0]

            # 事后检验
            sig_pairs = []
            if anova_p < 0.05:
                tukey = pairwise_tukeyhsd(temp_df[feature], temp_df['Symptom_Subtype'])
                sig_pairs = [f"{pair[0]}-{pair[1]}" for pair in tukey._results_table.data if pair[3]]

            # 计算效应量（每个亚型与总体）
            means = temp_df.groupby('Symptom_Subtype')[feature].mean()
            std_pooled = temp_df[feature].std()
            effect_sizes = ((means - temp_df[feature].mean()) / std_pooled).to_dict()

            results.append({
                'Feature': feature,
                'Feature_Group': group_name,
                'ANOVA_p': anova_p,
                'Significant_Pairs': sig_pairs,
                **{f'Mean_{k}': v for k, v in means.items()},
                **{f'Effect_{k}': v for k, v in effect_sizes.items()}
            })

    return pd.DataFrame(results)

# 5. 执行特征分析
symptom_feature_results = analyze_symptom_features(pd_data, feature_groups)

# 6. 保存分析结果
symptom_feature_results.to_csv('symptom_feature_analysis.csv', index=False)

# 7. 可视化分析
# 7.1 特征分布箱线图
plt.figure(figsize=(18, 12))
for i, feat in enumerate(selected_features[:12], 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x='Symptom_Subtype', y=feat, data=pd_data, order=symptom_subtypes)
    plt.title(feat)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('symptom_feature_distributions.png')
plt.close()

# 7.2 效应量热图（运动型 vs 非运动型）
motor_data = pd_data[pd_data['Symptom_Subtype'].isin(motor_subtypes)]
non_motor_data = pd_data[pd_data['Symptom_Subtype'].isin(non_motor_subtypes)]

effect_sizes = {}
for feat in selected_features:
    m_mean = motor_data[feat].mean()
    nm_mean = non_motor_data[feat].mean()
    pooled_std = np.sqrt((motor_data[feat].var() + non_motor_data[feat].var()) / 2)
    effect_sizes[feat] = (m_mean - nm_mean) / pooled_std

effect_df = pd.DataFrame.from_dict(effect_sizes, orient='index', columns=['Effect_Size'])

plt.figure(figsize=(12, 8))
sns.heatmap(effect_df.T, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            yticklabels=['Motor vs Non-Motor'])
plt.title('Motor vs Non-Motor Symptom Effect Sizes')
plt.savefig('symptom_effect_size_heatmap.png')
plt.close()

# 8. 生成特征签名
def generate_feature_signatures(results_df):
    signatures = {}

    for subtype in symptom_subtypes:
        # 筛选该亚型显著的特征
        subtype_features = []
        for _, row in results_df.iterrows():
            if any(subtype in pair for pair in row['Significant_Pairs']):
                effect = row[f'Effect_{subtype}']
                if abs(effect) > 0.5:  # 中等到大的效应量
                    subtype_features.append({
                        'feature': row['Feature'],
                        'effect_size': effect,
                        'mean': row[f'Mean_{subtype}']
                    })

        # 按效应量排序
        top_high = sorted(subtype_features, key=lambda x: x['effect_size'], reverse=True)[:3]
        top_low = sorted(subtype_features, key=lambda x: x['effect_size'])[:3]

        signatures[subtype] = {
            'High_Features': [{'Feature': f['feature'], 'Effect_Size': f['effect_size']} for f in top_high],
            'Low_Features': [{'Feature': f['feature'], 'Effect_Size': f['effect_size']} for f in top_low]
        }

    return signatures

feature_signatures = generate_feature_signatures(symptom_feature_results)

# 9. 保存结果
with open('symptom_feature_signatures.pkl', 'wb') as f:
    pickle.dump(feature_signatures, f)

with open('symptom_visualization_data.pkl', 'wb') as f:
    pickle.dump({
        'effect_sizes': effect_sizes,
        'feature_distributions': selected_features[:12],
        'motor_subtypes': motor_subtypes,
        'non_motor_subtypes': non_motor_subtypes
    }, f)

# 10. 生成报告
print("关键发现：")
print("运动型症状特征：")
print("待总结")

print("\n非运动型症状特征：")
print("待总结")

print("\n详细信息已保存至：symptom_feature_analysis.csv 和 symptom_feature_signatures.pkl")