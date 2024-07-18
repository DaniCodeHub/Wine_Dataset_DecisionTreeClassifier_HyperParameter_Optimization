# import libraries and modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
from sklearn.datasets import load_wine

load_wine()

data = load_wine()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
print(dataset)

X = dataset.copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

dtc = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=2, random_state=100, ccp_alpha=0)
dtc = dtc.fit(X_train, y_train)
print(dtc)

dtc.get_params()

dtc.predict_proba(X_train)

predictions = dtc.predict(X_train)
print(predictions)

recall_score(y_train, predictions, average='weighted')
precision_score(y_train, predictions, average='weighted')
confusion_matrix(y_train, predictions, labels=[0, 1, 2])
print(classification_report(y_train, predictions, target_names=['class_0', 'class_1', 'class_2']))

p_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'ccp_alpha': np.arange(0, 1, 0.1),
}
# Use GridSearchCV estimator to test parameters with cross validation value of 5
grid = GridSearchCV(dtc, param_grid=p_grid, cv=5)
print(grid)

grid.fit(X_train, y_train)
pd.DataFrame(grid.cv_results_).iloc[:,4:].sort_values('rank_test_score')

grid.best_estimator_
grid.best_score_

best_model = grid.best_estimator_

best_predictions = best_model.predict(X_train)
print(best_predictions)

recall_score(y_train, best_predictions, average='weighted')
precision_score(y_train, best_predictions, average='weighted')
confusion_matrix(y_train, best_predictions, labels=[0, 1, 2])
print(classification_report(y_train, best_predictions, target_names=['class_0', 'class_1', 'class_2']))

feature_names = X.columns
print(feature_names)

feature_importance = pd.DataFrame(best_model.feature_importances_, index=feature_names).sort_values(0, ascending=False)
print(feature_importance)

features = list(feature_importance[feature_importance[0]>0].index)
print(features)

feature_importance.head(13).plot(kind='bar')
plt.show()

fig = plt.figure(figsize=(25,25))
_ = tree.plot_tree(best_model,
                   feature_names=feature_names,
                   class_names={0: 'class_0', 1: 'class_1', 2: 'class_2'},
                   filled=True,
                   fontsize=12)
plt.show()
