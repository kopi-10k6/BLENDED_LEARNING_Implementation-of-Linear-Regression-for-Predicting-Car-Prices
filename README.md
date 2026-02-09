# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and read the car price dataset.
2. Define features (X) and target price (Y).
3. Split data into train/test and scale the features.
4. Train the Linear Regression model.
5. Predict test results and evaluate using MSE, RMSE, MAE, and R².
6. Plot graphs to verify linearity, independence, homoscedasticity, and normality.

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: KOPIGA N
RegisterNumber: 25011618 
*/
```
```

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head
X=df[['enginesize','horsepower','citympg','highwaympg']]
Y=df['price']
df.head()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#feature scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#train model
model=LinearRegression()
model.fit(X_train_scaled,Y_train)
#prediction
Y_pred=model.predict(X_test_scaled)
print("Name:Rubika")
print("Reg. No:212225040348")
print("MODEL COEFFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10}")
print(f"{'Intercept':>12}: {model.intercept_:>10}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(Y_test,Y_pred):>10}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(Y_test,Y_pred)):>10}")
print(f"{'R-squared':>12}: {r2_score(Y_test,Y_pred):10}")
print(f"{'MAE':>12}: {mean_absolute_error(Y_test,Y_pred):10}")
# linearity check
plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred,alpha=0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Price")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
# Independence (Durbin-watson)
residuals=Y_test-Y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicates no autocorrelation)")
# Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=Y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
# Normality of residuals
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distrubution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
<img width="1173" height="629" alt="Screenshot 2026-02-09 122855" src="https://github.com/user-attachments/assets/b27061e4-23ae-4219-9005-4c980f3b967f" />
<img width="1280" height="600" alt="Screenshot 2026-02-09 122928" src="https://github.com/user-attachments/assets/d7ec670e-3c29-4367-b943-7d7ccba70ed2" />
<img width="1404" height="556" alt="Screenshot 2026-02-09 122946" src="https://github.com/user-attachments/assets/687c4d90-95eb-4704-b505-2738f34629e8" />

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
