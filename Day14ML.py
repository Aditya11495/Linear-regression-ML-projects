    #Making a study and sleep hours based marks prediction model
    
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

data = ('StudyHours,SleepHours\n')
X = np.array([[2,4], [3,5], [5,6], [7,8], [8,9], [9,10], [4,5], [6,7], [9,8]])
Y = np.array([40, 50, 55, 65, 70, 80, 58, 62 ,87])

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)       # Train the model with training data  
y_pred = model.predict(X_test)   # Predict for test data
print("Actual:", Y_test)
print("Predicted:", y_pred)

print("Train R2:", r2_score(Y_train, model.predict(X_train)))
print("Test R2 :", r2_score(Y_test, y_pred))

score = model.score(X_test, Y_test)
print("Model Accuracy score:", score)


# Student studied 6 hours, slept 7 hours
prediction = model.predict([[6, 7]])
print("Predicted Marks:", prediction)