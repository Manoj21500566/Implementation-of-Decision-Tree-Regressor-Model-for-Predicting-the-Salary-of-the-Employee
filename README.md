# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
~~~
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Manoj M
RegisterNumber:  212221240027
*/
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
~~~

## Output:
Head:


![b1](https://user-images.githubusercontent.com/94588708/173001293-1198177f-29f8-4a55-965c-3faa6ca875f0.jpg)


Info:


![b2](https://user-images.githubusercontent.com/94588708/173001339-5611634f-97dc-44a9-8eb1-1445e3a597b3.jpg)

Isnull:

![b3](https://user-images.githubusercontent.com/94588708/173001401-6064f01d-50eb-4698-907c-b4b941160f7c.jpg)


Head using label encoder:



![b4](https://user-images.githubusercontent.com/94588708/173001488-48aace55-c088-43fa-b058-17ab5cdd4de1.jpg)

Mean square error:


![b5](https://user-images.githubusercontent.com/94588708/173001584-a87bd65a-5e8a-4393-9d9c-919c4b6b8efe.jpg)



r2:



![b6](https://user-images.githubusercontent.com/94588708/173001644-32610925-6c96-44e9-83d1-00471d23c322.jpg)

Array:




![A7](https://user-images.githubusercontent.com/94588708/173001695-187c2487-da73-4b35-8672-38cc3166c0f1.png)





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
