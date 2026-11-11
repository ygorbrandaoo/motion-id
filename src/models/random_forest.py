import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from har.data import load_dataset

dataset = load_dataset()
X_train = np.array(dataset.get("X_train"))
y_train = np.array(dataset.get("y_train"))
X_test = np.array(dataset.get("X_test"))
y_test = np.array(dataset.get("y_test"))
x_labels = np.array(dataset.get("feature_names"))
y_labels = np.array(dataset.get("activity_names"))

param_grid = {
    'n_estimators' : [100, 200],
    'max_features' : [None, "sqrt", "log2"],
    'max_depth' : [None, 5, 10],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(rf_model.score(X_test, y_test))
