import pandas as pd 
import numpy as np 
df = pd.read_csv("test_scores.csv")
import sklearn 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[['math']],df['cs'])
# print('m:',model.coef_)
# print('b:',model.intercept_)



