import numpy as np

def linearregression(x,y):
    # we have to find the value to m and b for these data points
    # y=mx+b
    steps = 1000
    step_size = 0.05 #learning rate
    m_current = b_current = 0
    n=len(x)
    i=0
    for i in range(steps):
        y_pred = m_current * x + b_current
        # print(type(y_pred))
        # print("predicted y is:",y_pred,"\n")
        md = -(2/n)* sum(x*(y-y_pred))
        bd = -(2/n)* sum((y-y_pred))
        # print("md:",md ," ,bd:",bd)
        m_current = m_current - step_size * md
        b_current = b_current - step_size * bd
        print("m:",m_current," b:",b_current," steps:",i+1)


x_arr = np.array([1,2,3,4,5])
y_arr = np.array([5,7,9,11,13])
linearregression(x_arr,y_arr)  


print("====comparing with sklearn====")
import sklearn 
from sklearn.linear_model import LinearRegression
import pandas as pd 
df = pd.read_csv("sample.csv")
model = LinearRegression()
model.fit(df[['x']],df['y'])
a = model.coef_
b = model.intercept_
print("slope:",a)
print("intercept:",b)


    



    

