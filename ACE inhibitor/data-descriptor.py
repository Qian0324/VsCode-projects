from modlamp.descriptors import *
import pandas as pd

#读取序列
df = pd.read_excel('E:\\SVM模型\\data.xlsx')
sequence = df['Sequence']
combined_df = pd.DataFrame()

#计算描述符
for data in sequence:
    desc = GlobalDescriptor(data)
    desc.calculate_all()
    df = pd.DataFrame(desc.descriptor, columns= desc.featurenames)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

#保存描述符
merged_df = pd.concat([sequence, combined_df], axis=1)
merged_df.to_excel('descriptor.xlsx')
print(merged_df)