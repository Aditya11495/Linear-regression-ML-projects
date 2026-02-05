#When Should You Use KNN?

# ✔ Small dataset
# ✔ Clear distance-based pattern
# ✔ Non-linear boundaries
# ✔ Baseline model comparison

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Dataset
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([0,0,0,0,1,1,1,1])  # 0 = Fail, 1 = Pass

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Prediction
prediction = model.predict([[2]])
print("Prediction:", prediction)
if prediction[0] == 0:
    
    print("Result: Fail")   
else:
    print("Result: Pass")
