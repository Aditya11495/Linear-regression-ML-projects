# Here we are testing and training a simple linear regression model using sklearn

import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)

Y = np.array([20,27,31,36,42,44,52,54,61,66])


X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2, random_state=42)
# here 0.2 means 20% data will be used for testing and 80% for training and  
# if (rows<1k then 0.2) use if (rows>1k then 0.3) and random _state can be 0,1 42 anything


model = LinearRegression()
model.fit(X_train, Y_train)       # Train the model with training data
y_pred = model.predict(X_test)   # Predict for test data

print("Actual:", Y_test)
print("Predicted:", y_pred)

score = model.score(X_test, Y_test)
print("Model Accuracy score:", score)






