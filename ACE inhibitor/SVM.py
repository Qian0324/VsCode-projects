import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,matthews_corrcoef
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,auc
from math import sqrt
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import joblib

# 数据集准备
df = pd.read_excel('descriptor.xlsx')
x = df.iloc[:, 3:-1]
y = df['label']


# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)

# 模型搭建与训练
clf = svm.SVC(kernel='linear', C=1.5, random_state=16, probability=True)
clf.fit(x_train, y_train)
svm_class_pred = clf.predict(x_test)
svm_proba_pred = clf.predict_proba(x_test)[:,1]


# 计算评估指标
cm = confusion_matrix(y_true=y_test, y_pred=svm_class_pred)
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

print("测试集上的准确率(Pr)：", end=' ')
print((TN+TP)/(TN+FP+FN+TP))

print("测试集上的灵敏度(Sn)：", end=' ')
print(TP/(TP+FN))

print("测试集上的特异性(Sp)：", end=' ')
print(TN/(TN+FP))

print("测试集上的马修相关系数（MCC）：", end=' ')
print((TP*TN-FP*FN)/sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

print("SVM AUC:",roc_auc_score(y_test ,svm_proba_pred))

#可视化分析（混淆矩阵）
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, svm_proba_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC crve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 模型保存与调用
joblib.dump(clf, 'svc.pkl')
model = joblib.load('svc.pkl')