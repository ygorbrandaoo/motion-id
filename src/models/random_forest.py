import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from har.data import load_dataset, load_activity_names, load_feature_names, load_split

dataset = load_dataset()
X_train = np.array(dataset.get("X_train"))
y_train = np.array(dataset.get("y_train"))
X_test = np.array(dataset.get("X_test"))
y_test = np.array(dataset.get("y_test"))
x_labels = np.array(dataset.get("feature_names"))
y_labels = np.array(dataset.get("activity_names"))

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(rf.score(X_test, y_test))
