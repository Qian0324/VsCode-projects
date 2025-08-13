import math
import joblib
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#数据准备与打乱
random.seed(42)
tf.random.set_seed(42)
df = pd.read_excel('E:\VsCode-projects\ACE inhibitor\descriptor.xlsx')
x = df.iloc[:, 3:-1].values
y = df['label'].values
#标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split( x_scaled, y, test_size=0.2, random_state=8, stratify=y)
#将标签转化为独热编码，适应softmax识别
y_train_tf = keras.utils.to_categorical(y_train, 2)
y_test_tf = keras.utils.to_categorical(y_test, 2)

#数据转化成1D CNN数据
feature_count = x_train.shape[1]
x_train_reshaped = x_train.reshape(-1,feature_count,1)
x_test_reshaped = x_test.reshape(-1,feature_count,1)

#1D modle
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=64, kernel_size=(3),activation="relu",input_shape=(feature_count,1),padding="same"))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.MaxPooling1D(pool_size=(2), padding="same"))
model.add(keras.layers.Conv1D(filters=32, kernel_size=(3),activation="relu",padding="same"))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.MaxPooling1D(pool_size=(2),padding="same"))
model.add(keras.layers.Conv1D(filters=16, kernel_size=(3),activation="relu",padding="same"))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=64,activation="relu"))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.Dense(units=2,activation="softmax"))

#模型结构
model.summary()

#编译模型
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=["accuracy"]
)
epochs =30
batch_size =15
validation_split = 0.2

# 回调函数 - 记录每个epoch的预测
class RecordPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs):
        super(RecordPredictionsCallback, self).__init__()
        self.inputs = inputs
        self.predictions = []

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.inputs)
        self.predictions.append(predictions)

# 创建回调实例
record_predictions_callback = RecordPredictionsCallback(x_test_reshaped)

#拟合模型
history = model.fit(
    x_train_reshaped, y_train_tf,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=[record_predictions_callback],
    verbose=1
)

#获取训练过程中的损失和准确率
loss = history.history['loss']
accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

#模型保存与预处理器
model.save('ACE_inhibitory_CNN_model.h5')
joblib.dump(scaler, 'ACE_inhibitory_CNN_model.pkl')

# visualization training
plt.figure(figsize=(16, 6))
#创建损失函数图表
plt.subplot(1,2,1)
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
#创建准确率图表
plt.subplot(1,2,2)
plt.plot(accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
#保存图片为PDF
plt.savefig("training_curves.pdf", format='pdf', bbox_inches='tight')
plt.show()

# predict
#预测测试集的概率
pred_tf = model.predict(x_test_reshaped)
pred_tf_df = pd.DataFrame(pred_tf,columns=["nb_prob","b_prob"])

#真实标签
y_real_tf = np.argmax(y_test_tf,axis=1)
#预测标签
y_pred_tf = np.argmax(pred_tf,axis=1)
#模型评估
perf_tf = model.evaluate(x_test_reshaped, y_test_tf, verbose=0)
acc_tf = round(perf_tf[1] * 100, 3)

# 创建结果数据框
results_tf = pd.DataFrame({
    "y_real": y_real_tf,
    "y_pred": y_pred_tf,
    "b_prob": pred_tf_df["b_prob"]
})

# TP/FP/TN/FN 
#添加类别标签
results_tf["class"] = np.where((results_tf["y_real"] == 0) & (results_tf["y_pred"] == 0),"TN",
                               np.where((results_tf["y_real"] ==0 ) & (results_tf["y_pred" == 1]),"FP",
                                        np.where((results_tf["y_real" == 1]) & (results_tf["y_pred" == 1]),"TP",
                                                 np.where((results_tf["y_real" == 1]) & (results_tf["y_pred" == 0]),"FN",np.nan)
                                                 )
                                        )
                               )
#生成混淆矩阵
res_tf = pd.crosstab(results_tf["y_real"],results_tf["y_pred"]) #.reset_index()
X = res_tf.values
res_tf = pd.DataFrame({
    'y_real':['Non-inhibitor','Inhibitor','Non-inhibitor','Inhibitor'],
    'y_pred':['Non-inhibitor','Non-inhibitor','Inhibitor','Inhibitor'],
    'count':[X[0,0],X[1,0],X[0,1],X[1,1]],
    'class':["TN","FN","FP","TP"]
})

TN = float(res_tf[res_tf['class'] == 'TN']['count'])
TP = float(res_tf[res_tf['class'] == 'TP']['count'])
FN = float(res_tf[res_tf['class'] == 'FN']['count'])
FP = float(res_tf[res_tf['class'] == 'FP']['count'])

Accuracy = (TN + TP) / (TN + TP + FN + FP)
Sensitivity = TP / (TP + FN)  # Recall
Specificity = TN / (TN + FP)
Precision = TP / (TP + FP)
F1_Score = 2 * Precision * Sensitivity / (Precision + Sensitivity)
MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
roc_auc = roc_auc_score(y_real_tf, results_tf["b_prob"])

print("详细评估指标:")
print(f"准确率 (Accuracy): {Accuracy:.4f}")
print(f"灵敏度 (Sensitivity/Recall): {Sensitivity:.4f}")
print(f"特异性 (Specificity): {Specificity:.4f}")
print(f"精确度 (Precision): {Precision:.4f}")
print(f"F1分数: {F1_Score:.4f}")
print(f"马修相关系数 (MCC): {MCC:.4f}")
print(f"AUC: {roc_auc:.4f}")

# 绘制混淆矩阵
confusion_matrix = res_tf.pivot('y_real', 'y_pred', 'count')
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ACE Inhibitor Prediction")
plt.suptitle(f"Accuracy = {acc_tf}%")
plt.savefig("confusion_matrix.pdf", format='pdf', bbox_inches='tight')
plt.show()
#绘制密度图
plt.figure(figsize=(8, 6))
g = sns.FacetGrid(results_tf, hue="class")
g.map(sns.kdeplot, "b_prob")
g.add_legend()
plt.xlabel("Inhibitor probability")
plt.ylabel("Density")
plt.title("ACE Inhibitor Probability Distribution")
plt.savefig("probability_density.pdf", format='pdf', bbox_inches='tight')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_real_tf, results_tf["b_prob"])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC curve (AUC = {:.2f})".format(roc_auc))
plt.plot([0, 1], [0, 1], color="black", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ACE Inhibitor ROC curve")
plt.legend(loc="lower right")
plt.savefig("CNN_roc_curve.pdf", format='pdf', bbox_inches='tight')
plt.show()

# 打印每个epoch的预测结果分析
score_1_train = list()
precision_1_train = list()
recall_1_train = list()
accuracy_train = list()

for epoch in range(epochs):
    epoch_predictions = record_predictions_callback.predictions[epoch]
    threshold = 0.5
    binary_predictions = np.where(epoch_predictions > threshold, 1, 0)
    y_real_epoch = np.argmax(y_test_tf, axis=1)
    y_pred_epoch = np.argmax(binary_predictions, axis=1)
    
    results_epoch = pd.DataFrame({"y_real": y_real_epoch, "y_pred": y_pred_epoch})
    
    # 添加类别
    results_epoch["class"] = np.where((results_epoch["y_real"] == 0) & (results_epoch["y_pred"] == 0), "TN",
                                     np.where((results_epoch["y_real"] == 0) & (results_epoch["y_pred"] == 1), "FP",
                                             np.where((results_epoch["y_real"] == 1) & (results_epoch["y_pred"] == 1), "TP",
                                                     np.where((results_epoch["y_real"] == 1) & (results_epoch["y_pred"] == 0), "FN", np.nan)
                                                     )
                                             )
                                     )
    
    res_epoch = pd.crosstab(results_epoch["y_real"], results_epoch["y_pred"])
    X_epoch = res_epoch.values

    #检查矩阵维度
    if X_epoch.shape[1] > 1 and X_epoch.shape[0] > 1:
        TN_epoch = float(X_epoch[0,0])
        TP_epoch = float(X_epoch[1,1])
        FN_epoch = float(X_epoch[1,0])
        FP_epoch = float(X_epoch[0,1])
        
        Accuracy_epoch = (TN_epoch + TP_epoch) / (TN_epoch + TP_epoch + FN_epoch + FP_epoch)
        Precision_epoch = TP_epoch / (TP_epoch + FP_epoch) if (TP_epoch + FP_epoch) > 0 else 0
        Recall_epoch = TP_epoch / (TP_epoch + FN_epoch) if (TP_epoch + FN_epoch) > 0 else 0
        F1_epoch = 2 * Precision_epoch * Recall_epoch / (Precision_epoch + Recall_epoch) if (Precision_epoch + Recall_epoch) > 0 else 0
        
        score_1_train.append(F1_epoch)
        precision_1_train.append(Precision_epoch)
        recall_1_train.append(Recall_epoch)
        accuracy_train.append(Accuracy_epoch)
        
        print(f"Epoch {epoch+1}: Accuracy = {Accuracy_epoch:.4f}, Precision = {Precision_epoch:.4f}, Recall = {Recall_epoch:.4f}, F1 = {F1_epoch:.4f}")
    else:
        print(f"Epoch {epoch+1}: 预测结果单一，无法计算完整指标")
        score_1_train.append(0)
        precision_1_train.append(0)
        recall_1_train.append(0)
        accuracy_train.append(0)

# 绘制训练过程指标变化图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(score_1_train, label='F1 Score')
plt.plot(precision_1_train, label='Precision')
plt.plot(recall_1_train, label='Recall')
plt.title('Training Metrics per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_train, label='Accuracy per Epoch', color='green')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("epoch_metrics.pdf", format='pdf', bbox_inches='tight')
plt.show()

#保存训练结果
results_tf.to_csv('detailed_predictions.csv', index=False)

# BLOSUM62 ENCODE <-2D CNN     # 8-40肽长
# 如需使用2D CNN，准备BLOSUM62矩阵
# 1. 下载BLOSUM62矩阵文件 (e.g.:matrix_data.csv)
# 2. 使用pep_encode_blosum函数对肽序列进行编码
# 3. 重塑数据为(samples, length, 22, 1)格式用于2D CNN
BLOSUM62 = pd.read_csv("matrix_data.csv",index_col=0)  #""中填入对应的BLAST后所得到的.csv文件
#定义编码函数
def pep_encode_blosum(pep):
    """
    将肽序列编码为BLOSUM62矩阵表示
    输入：肽序列的pandas Series
    输出：编码后的3D数组(n_peptides,peptide_length,22)
    """
    p_mat = np.array([list(peptide) for peptide in pep])
    n_peps = len(pep)
    l_peps = len(pep.iloc[0])
    l_enc = len(BLOSUM62)
    o_tensor = np.empty((n_peps,l_peps,l_enc))

    for i in range(n_peps):
        pep_i_residues = p_mat[i,:]
        pep_i_residues = pep_i_residues.tolist()
        pep_img = BLOSUM62.loc[pep_i_residues]
        o_tensor[i,:,:] = pep_img
    return o_tensor

#对数据进行编码
tf_train = pep_encode_blosum(x_train)
tf_test = pep_encode_blosum(x_test)
tf_train = tf_train.reshape(tf_train.shape[0], tf_train.shape[1], 22, 1)
tf_test = tf_test.reshape(tf_test.shape[0], tf_test.shape[1], 22, 1)

def create_2d_cnn_model(input_shape):
    """
    input_shape: (length, 22, 1) - 序列长度, BLOSUM62特征数, 通道数
    """
    model_2d = keras.models.Sequential()
    model_2d.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", 
                                   input_shape=input_shape, padding="same"))
    model_2d.add(keras.layers.Dropout(rate=0.3))
    model_2d.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same"))
    model_2d.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
    model_2d.add(keras.layers.Dropout(rate=0.4))
    model_2d.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same"))
    model_2d.add(keras.layers.Flatten())
    model_2d.add(keras.layers.Dense(units=64, activation="relu"))
    model_2d.add(keras.layers.Dropout(rate=0.5))
    model_2d.add(keras.layers.Dense(units=2, activation="softmax"))
    
    return model_2d

# 准备标签
y_train_2d = keras.utils.to_categorical(tf_train, 2)
y_test_2d = keras.utils.to_categorical(tf_test, 2)

# 创建2D CNN模型
model_2d = create_2d_cnn_model((40, 22, 1))
model_2d.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#2D模型结构
model_2d.summary()

# 创建2D CNN的回调函数
record_predictions_callback_2d = RecordPredictionsCallback(tf_test)

#拟合模型
history_2d = model_2d.fit(
    tf_train, y_train_2d,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=[record_predictions_callback_2d],
    verbose=1
)

#保存训练好的模型
model_2d.save('ACE_inhibitory_2D_CNN_model.h5')
joblib.dump(scaler, 'ACE_inhibitory_2D_CNN_model.pkl')

# predict
pred_2d = model_2d.predict(tf_test)
pred_2d_df = pd.DataFrame(pred_2d, columns=["nb_prob_2d", "b_prob_2d"])

y_real_2d = np.argmax(y_test_2d, axis=1)
y_pred_2d = np.argmax(pred_2d, axis=1)

perf_2d = model_2d.evaluate(tf_test, y_test_2d, verbose=0)
acc_2d = round(perf_2d[1] * 100, 3)

# 创建结果数据框
results_2d = pd.DataFrame({
    "y_real": y_real_2d,
    "y_pred": y_pred_2d,
    "b_prob": pred_2d_df["b_prob"]
})
 
#添加类别标签
results_2d["class"] = np.where((results_2d["y_real"] == 0) & (results_2d["y_pred"] == 0),"TN",
                               np.where((results_2d["y_real"] ==0 ) & (results_2d["y_pred" == 1]),"FP",
                                        np.where((results_2d["y_real" == 1]) & (results_2d["y_pred" == 1]),"TP",
                                                 np.where((results_2d["y_real" == 1]) & (results_2d["y_pred" == 0]),"FN",np.nan)
                                                 )
                                        )
                               )
#生成混淆矩阵
res_2d = pd.crosstab(results_2d["y_real"],results_2d["y_pred"]) #.reset_index()
X_2d = res_2d.values

TN_2d = float(X_2d[0,0])
TP_2d = float(X_2d[1,1])
FN_2d = float(X_2d[1,0])
FP_2d = float(X_2d[0,1])

Accuracy_2d = (TN_2d + TP_2d) / (TN_2d + TP_2d + FN_2d + FP_2d)
Sensitivity_2d = TP_2d / (TP_2d + FN_2d)
Specificity_2d = TN_2d / (TN_2d + FP_2d)
Precision_2d = TP_2d / (TP_2d + FP_2d)
F1_Score_2d = 2 * Precision * Sensitivity / (Precision + Sensitivity)
MCC_2d = (TP_2d * TN_2d - FP_2d * FN_2d) / math.sqrt((TP_2d + FP_2d) * (TP_2d + FN_2d) * (TN_2d + FP_2d) * (TN_2d + FN_2d))
roc_auc_2d = roc_auc_score(y_real_2d, results_2d["b_prob"])

print(f"准确率 (Accuracy): {Accuracy_2d:.4f}")
print(f"灵敏度 (Sensitivity/Recall): {Sensitivity_2d:.4f}")
print(f"特异性 (Specificity): {Specificity_2d:.4f}")
print(f"精确度 (Precision): {Precision_2d:.4f}")
print(f"F1分数: {F1_Score_2d:.4f}")
print(f"马修相关系数 (MCC): {MCC_2d:.4f}")
print(f"AUC: {roc_auc_2d:.4f}")

#获取训练过程中的损失和准确率
loss_2d = history_2d.history['loss']
accuracy_2d = history_2d.history['accuracy']
val_loss_2d = history_2d.history['val_loss']
val_accuracy_2d = history_2d.history['val_accuracy']

plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
plt.plot(loss_2d, label = '2D CNN Training Loss')
plt.plot(val_loss_2d, label = '2D CNN Validation Loss')
plt.title('2D CNN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(accuracy_2d, label = '2D CNN Training Accuracy')
plt.plot(val_accuracy_2d, label = '2D CNN Validation Accuracy')
plt.title('2D CNN Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("2D_CNN_training_curves.pdf", format='pdf', bbox_inches='tight')
plt.show()

# ROC曲线对比
fpr_2d, tpr_2d, _ = roc_curve(y_real_2d, results_2d["b_prob"])

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f"1D CNN (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot(fpr_2d, tpr_2d, label=f"2D CNN (AUC = {roc_auc_2d:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], color="black", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: 1D CNN vs 2D CNN")
plt.legend(loc="lower right")
plt.savefig("CNN_comparison_roc.pdf", format='pdf', bbox_inches='tight')
plt.show()

#保存训练结果
results_2d.to_csv('2D_CNN_detailed_predictions.csv', index=False)

# 模型比较表
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'MCC', 'AUC'],
    '1D CNN': [Accuracy, Sensitivity, Specificity, Precision, F1_Score, MCC, roc_auc],
    '2D CNN': [Accuracy_2d, Sensitivity_2d, Specificity_2d, Precision_2d, F1_Score_2d, MCC_2d, roc_auc_2d]
})

print(comparison_df.round(4))
comparison_df.to_csv('model_comparison.csv', index=False)