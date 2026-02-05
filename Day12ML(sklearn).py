import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array([30,35,40,45,50,55,60,65]).reshape(-1,1)

Y = np.array([1,2,2,3,4,4,5,6])

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.20, random_state=1)
model = LinearRegression()
model.fit(X_train, Y_train)       # Train the model with training data
y_pred = model.predict(X_test)   # Predict for test data
print("Actual:", Y_test)
print("Predicted:", y_pred)

score = model.score(X_test, Y_test)

prediction = model.predict(np.array([[25]]))  #Want to check for 25 degree for checking model working or not

print('prediction for 25 deg:',prediction)

print("Model Accuracy score:", score) #

import matplotlib.pyplot as plt
plt.scatter(X, Y, color='blue', label='Actual Data') # 

plt.plot(X, model.predict(X), color='red', label='Regression Line')

plt.scatter(X, model.predict(X), color='red', label='Predicted')

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
