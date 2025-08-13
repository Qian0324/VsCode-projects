import pandas as pd

df = pd.read_excel('descriptor.xlsx')
print("DataFrame 中的列名:", df.columns) # 添加这一行
x = df.iloc[:, 3:-1]
