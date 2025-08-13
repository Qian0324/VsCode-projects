import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,confusion_matrix,roc_curve,auc
from math import sqrt
import joblib

# 数据准备
df = pd.read_excel('E:\VsCode-projects\ACE inhibitor\descriptor.xlsx')
x = df.iloc[:, 3:-1]
y = df['label']


# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)


# 模型训练
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_class_pred = rf_model.predict(x_test)
rf_proba_pred = rf_model.predict_proba(x_test)[:,1]


# 模型评估
cm = confusion_matrix(y_true=y_test, y_pred=rf_class_pred)
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

print("测试集上的准确率(Pr)：", end=' ')
print((TN+TP)/(TN+FP+FN+TP))

print("测试集上的灵敏度(Sn)：", end=' ')
print(TP/(TP+FN))

print("测试集上的特异性(Sp)：", end=' ')
print(TN/(TN+FP))

print("测试集上的马修相关系数（MCC）：", end=' ')
print((TP*TN-FP*FN)/sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

print("Random Forest AUC:",roc_auc_score(y_test ,rf_proba_pred))


#可视化分析(混淆矩阵)
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, rf_class_pred), annot=True, fmt='d', cmap='Blues')
plt.title("RF Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, rf_proba_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC crve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF ROC Curve')
plt.legend(loc='lower right')
plt.show()


# 模型保存与调用
joblib.dump(rf_model, 'Random_Forest.pkl')
model = joblib.load('Random_Forest.pkl')