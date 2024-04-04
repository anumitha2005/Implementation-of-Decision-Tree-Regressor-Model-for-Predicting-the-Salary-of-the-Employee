# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
``
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: anumitha.m.r
RegisterNumber: 
*/
```
```
import pandas as pd
d=pd.read_csv("Salary.csv")
d.head()
d.info()
d.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
d["Position"] = l.fit_transform(d["Position"])
d.head()

x = d[["Position","Level"]]
y = d["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
``
``
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
Head:

![image](https://github.com/anumitha2005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/155522855/36c72000-2036-4ea0-ba63-7254935e2a48)

Info:

![image](https://github.com/anumitha2005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/155522855/84233cdf-41ca-4fea-a80d-f0b8affb5d3e)

Isnull:

![image](https://github.com/anumitha2005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/155522855/652ccbfb-48b3-4aae-bf75-126c8e9ad9a5)

Head using label encoder:

![image](https://github.com/anumitha2005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/155522855/0568d7c3-4784-49fa-9d13-09acfd8e7ea0)

Mean square error:

![image](https://github.com/anumitha2005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/155522855/bc3ccc0c-be8d-4217-950d-6ad5f1cf3fe3)

r2:

![image](https://github.com/anumitha2005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/155522855/118507d9-63c3-4a3b-a1c2-984a301efa15)

Array:

![image](https://github.com/anumitha2005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/155522855/f366d268-b388-4bc4-9a78-5e5812347f1c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
