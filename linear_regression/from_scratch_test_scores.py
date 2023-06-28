import pandas as pd
import numpy as np
df = pd.read_csv("test_scores.csv")
x = df['math'].to_numpy()
y = df['cs'].to_numpy()
 
i=0
#OPTIMUM VALUES ARE
# steps = 100000
# step_size = 0.01
# m: 0.8995983935743249 b: 2.112449799196607 cost: 0.6248995983935746
n= len(x)

steps = 100000
step_size = 0.01
# when we do linear regression through sklearn, we obtain:
# m: [0.89959839]
# b: 2.1124497991967903

def gradientdescent(x,y):
    m = b = 0
    for i in range(steps):
        y_pred = m * x + b 
        cost = 1/n * sum((y-y_pred)**2)
        dm = -2/n * sum(x*(y-y_pred))
        db = -2/n * sum(y-y_pred)
        m = m - step_size * dm
        b = b - step_size * db
    print("m:",m,"b:",b, "cost:",cost)

gradientdescent(x,y)
