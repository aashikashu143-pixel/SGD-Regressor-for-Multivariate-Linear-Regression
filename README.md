# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: AASHIK A
RegisterNumber:  25012808
*/
```
```

from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt


X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])


model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')
model.fit(X, y)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)


y_pred = model.predict(X)
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  
plt.show()

```

## Output:
<img width="341" height="49" alt="Screenshot 2026-02-02 151426" src="https://github.com/user-attachments/assets/ab82b346-c023-49e4-a833-22c14938e249" />
<img width="779" height="574" alt="Screenshot 2026-02-02 151442" src="https://github.com/user-attachments/assets/baa7119c-ddf9-4f4a-ab0a-7e175a96c029" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
