import pandas as pd
import joblib

df = pd.read_excel('descriptor.xlsx')
x = df.iloc[:, 3:-1]

# 加载K-NN模型和标准化器
model = joblib.load('knn.pkl')
scaler = joblib.load('scaler.pkl')

# 数据标准化（使用训练时的标准化器）
x_scaled = scaler.transform(x)

# 预测
pred = model.predict(x_scaled)
print("K-NN预测结果：")
print(pred)