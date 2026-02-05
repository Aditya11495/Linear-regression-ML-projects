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

print("Actual:", Y_test)            #taking output
print("Predicted:", y_pred)

score = model.score(X_test, Y_test)   # Model accuracy

prediction = model.predict(np.array([[25]]))  

from sklearn.metrics import r2_score

train_pred = model.predict(X_train)   # Predict for train data
test_pred = model.predict(X_test)    # Predict for test data

print("Train R2:", r2_score(Y_train, train_pred))   # Train  score
print("Test R2 :", r2_score(Y_test, test_pred))     # Test  score