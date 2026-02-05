        #Logistic Regression Example(Yes/No or Pass/Fail)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Study hours
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)

# Result: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

study_hours = [[5]]
prediction = model.predict(study_hours)
probability = model.predict_proba(study_hours)

print("Prediction:", prediction)
print("Probability:", probability)

        #Visualizing the logistic regression curve

X_test = np.linspace(0, 9, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

plt.scatter(X, y, color='red')
plt.plot(X_test, y_prob)
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.show()



