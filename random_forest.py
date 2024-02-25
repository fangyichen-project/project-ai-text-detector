#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:14:45 2024

apply random forest and see how the results look like.

@author: fangyichen
"""
# %%

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.tree import export_graphviz
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from scipy.sparse import vstack
from sklearn.feature_selection import SelectFromModel

# %%
# Import x_train_tfidf, x_test_tfidf, y_train, y_test, x_train_df, features, x, y.
import text_tfidf_vectorizing
x_train_tfidf, x_test_tfidf, y_train, y_test, x_train_df, features, x, y = text_tfidf_vectorizing.text_tfidf_vectorizing()

# %%
#apply random forest.
rf = RandomForestClassifier()
rf = rf.fit(x_train_tfidf, y_train)

#Do prediction.
y_pred = rf.predict(x_test_tfidf)

# %%
# Get feature importances
feat_importances = pd.Series(rf.feature_importances_, index=features)

# Sort the feature importances by most important first
feat_importances_sorted = feat_importances.sort_values(ascending=False)

print(feat_importances_sorted[:50])
# %%
#Check the most important 20 features.
feat_importances.nlargest(20).plot(kind='bar',figsize=(10,10))
plt.title("Top 20 important features")
plt.show()

# %%
#Select the important features using selectfrommodel.
model = SelectFromModel(rf, prefit=True)
x_new_train = model.transform(x_train_tfidf)
x_test_tfidf = model.transform(x_test_tfidf)

# %%
#Hyperparameter optimization using Random Search Cross Validation.

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf= RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_seletected_feature_hypertune = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_seletected_feature_hypertune.fit(x_new_train, y_train)

print(rf_seletected_feature_hypertune.best_params_)

'''
{'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
'''


# %%
#Do prediction.

rf_seletected_feature_hypertune = RandomForestClassifier(
    n_estimators=800,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    max_depth=None,
    bootstrap=False
)

rf_seletected_feature_hypertune = rf_seletected_feature_hypertune.fit(x_new_train, y_train)
y_pred = rf_seletected_feature_hypertune.predict(x_test_tfidf)


# %%
#Evaluate the results.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of model applied random grid and hyperparameter optimization:", accuracy)

# %%
#View confusion matrix for test data and predictions
print("Confusion matrix:", confusion_matrix(y_test, y_pred))

# %%
## View the classification report for test data and predictions
print(classification_report(y_test, y_pred))


# %%
# using 5-cross-fold validation.
# It shows how well the model performed on each subset of the data.
scores = cross_val_score(rf,x,y,cv=5)

# Print individual fold scores
print("Cross-validation scores:", scores)
#The model performs quite balanced in all subsets of the data.

# Print the mean score across all folds
#It provides a single metric to summarize the overall performance of the model.
print("Mean Cross-validation score:", scores.mean())

print("Std of Cross-validation score:", scores.std())


# %%
## View the classification report for test data and predictions
print(classification_report(y_test, y_pred))

# %%    
#ROC u. AUC
roc_value = roc_auc_score(y_test, y_pred)

print("ROC value is",roc_value)

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


