import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from math import sqrt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 数据集准备
df = pd.read_excel('descriptor.xlsx')
x = df.iloc[:, 3:-1]
y = df['label']

# 数据标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=8)

# 模型搭建与训练
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')  # weights='uniform'表示所有邻居权重相等，'distance'表示距离越近权重越大
knn_clf.fit(x_train, y_train)
knn_class_pred = knn_clf.predict(x_test)
knn_proba_pred = knn_clf.predict_proba(x_test)[:,1]

# 计算评估指标
cm = confusion_matrix(y_true=y_test, y_pred=knn_class_pred)
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

print("测试集上的准确率(Pr)：", end=' ')
print((TN+TP)/(TN+FP+FN+TP))

print("测试集上的灵敏度(Sn)：", end=' ')
print(TP/(TP+FN))

print("测试集上的特异性(Sp)：", end=' ')
print(TN/(TN+FP))

print("测试集上的马修相关系数（MCC）：", end=' ')
print((TP*TN-FP*FN)/sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

print("K-NN AUC:",roc_auc_score(y_test, knn_proba_pred))

# 可视化分析（混淆矩阵）
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.title('K-NN Confusion Matrix')
plt.show()

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, knn_proba_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-NN ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 模型保存与调用
joblib.dump(knn_clf, 'knn.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 加载模型和标准化器
model = joblib.load('knn.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# K-NN超参数优化（可选）
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# 网格搜索
print("\n正在进行超参数优化...")
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(x_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证得分:", grid_search.best_score_)

# 使用最佳参数的模型
best_knn = grid_search.best_estimator_
best_knn_pred = best_knn.predict(x_test)
best_knn_proba = best_knn.predict_proba(x_test)[:,1]

print("优化后K-NN AUC:", roc_auc_score(y_test, best_knn_proba))

# 特征重要性分析（K-NN没有直接的特征重要性，但可以通过排列重要性分析）
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(knn_clf, x_test, y_test, n_repeats=10, random_state=42)

# 绘制特征重要性
feature_names = df.columns[3:-1]  # 获取特征名称
n_features = len(feature_names)
n_top_features = min(10, n_features)  # 取特征数量和10中的较小值

indices = perm_importance.importances_mean.argsort()[-n_top_features:][::-1]

plt.figure(figsize=(10, 6))
plt.title(f"K-NN Feature Importance (Top {n_top_features})")
plt.bar(range(n_top_features), perm_importance.importances_mean[indices])
plt.xticks(range(n_top_features), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()