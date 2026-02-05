#High Train data + Low Test data  → Overfitting
#Low Train + Low Test   → Under fitting
#Similar & High         → Best Model(train R2 ~test R2 + High)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Original useful features
X = np.array([
    [2,4],
    [3,5],
    [4,5],
    [5,6],            #study hours, sleep hours
    [6,7],
    [7,8],
    [8,9],
    [9,8]
])

Y = np.array([40,50,55,58,62,65,70,75])  #marks

X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

print("Train R2:", r2_score(Y_train, model.predict(X_train)))
print("Test  R2:", r2_score(Y_test, model.predict(X_test)))

        #output
#Here model is good and stable as both Train R2 and Test R2 are similar and high

            
            #Now Overfitting Example
            
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(0)

random_feature = np.random.randint(1, 100, size=len(X)) # Useless random feature

# New X with useless feature

X_bad = np.column_stack((X, random_feature))
X_train, X_test, Y_train, Y_test = train_test_split(
    X_bad, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

print("\nOutput With Useless Feature Overfitting:")

print("Train R2:", r2_score(Y_train, model.predict(X_train)))
print("Test  R2:", r2_score(Y_test, model.predict(X_test)))
