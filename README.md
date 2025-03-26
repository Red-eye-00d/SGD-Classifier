# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries and Load Dataset: Import necessary libraries (pandas, sklearn, matplotlib, seaborn) and load the Iris dataset using load_iris().

2.Create DataFrame: Convert the loaded Iris dataset into a pandas DataFrame with feature names as columns and target values.

3.Split Data: Separate the dataset into feature variables (X) and target variable (y), then split the data into training and testing sets using train_test_split().

4.Train Classifier: Initialize an SGDClassifier with specified parameters (max_iter=1000, tol=1e-3), and fit the classifier on the training data.

5.Evaluate Model: Predict target values for the test set, calculate accuracy using accuracy_score(), and display the confusion matrix with seaborn heatmap for better visualization.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SUDARSHANAN U
RegisterNumber: 212224230276

import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
iris=load_iris() 
df=pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['target']=iris.target 
print(df.head())
X = df.drop('target',axis=1) 
y=df['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}") 
cm=confusion_matrix(y_test,y_pred) 
print("Confusion Matrix:") 
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

*/
```

## Output:
![prediction of iris species using SGD Classifier](sam.png)
![Screenshot 2025-03-25 183307](https://github.com/user-attachments/assets/a8dd5b77-6a4d-464a-bf5b-ff65b0539e01)
![Screenshot 2025-03-25 183312](https://github.com/user-attachments/assets/bd299c70-480a-4e33-a74b-bc06d1eb7f77)
![Screenshot 2025-03-25 183327](https://github.com/user-attachments/assets/e08cb0b8-2e2e-4550-a4e5-7051cd8ff9f1)
![Screenshot 2025-03-25 183318](https://github.com/user-attachments/assets/5be44323-49b2-4300-99f9-5fea44cebc5a)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
