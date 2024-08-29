# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
  
## Algorithm
Step 1: Start.

Step 2: Import the standard Libraries.

Step 3: Set variables for assigning dataset values.

Step 4: Import linear regression from sklearn.

Step 5: Assign the points for representing in the graph.

Step 6: Predict the regression for marks by using the representation of the graph.

Step 7: Compare the graphs and hence we obtained the linear regression for the given datas.

Step 8: End.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: LAVANYA S

RegisterNumber: 212223230112 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## DataSet:
![image](https://github.com/user-attachments/assets/5ff33b39-210d-4766-b3aa-5e3473dff374)
## Head Values:
![image](https://github.com/user-attachments/assets/f501445a-e9c4-432b-af9b-30873f36fec3)
## Tail Values:
![image](https://github.com/user-attachments/assets/2141027a-19e3-4dba-9421-3484bef1ac9d)
## X and Y Values:
![image](https://github.com/user-attachments/assets/6effc881-f9fc-42bf-ade0-b3c2ca303b26)
## Prediction of X and Y:
![image](https://github.com/user-attachments/assets/66cb9040-9730-4a70-ac94-21303b0d9e78)
## MSE, MAE and RMSE:
![image](https://github.com/user-attachments/assets/bc9e53aa-29ae-463c-a750-7ebc8c12592a)
## Training Set:
![image](https://github.com/user-attachments/assets/eae8ee8f-f4bc-466a-a4f1-b7d464efe8e0)
## Testing Set:
![image](https://github.com/user-attachments/assets/185ead62-081f-4570-a47f-1ab80015077e)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

