import pandas as pd
import joblib

df = pd.read_excel('descriptor.xlsx')
x = df.iloc[:, 3:-1]

model = joblib.load('svc.pkl')
pred = model.predict(x)
print(pred)