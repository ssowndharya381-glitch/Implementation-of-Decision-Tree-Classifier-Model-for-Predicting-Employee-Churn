# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Load employee data and split it into training and testing sets.
Train a Decision Tree classifier using entropy as the split criterion.
Evaluate the model using accuracy, confusion matrix, and classification report.
Use the trained model to predict whether a new employee will stay or leave.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SOWNDHARYA S 
Register Number:  212225220100
*/
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Employee.csv")
df.rename(columns={'Departments ': 'Department'}, inplace=True)
df = pd.get_dummies(df, columns=['Department', 'salary'])
X = df.drop('left', axis=1)
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(15, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)

plt.show()
```

## Output:
<img width="483" height="402" alt="image" src="https://github.com/user-attachments/assets/a2997526-c6e7-42c8-acea-80c1d9497448" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
