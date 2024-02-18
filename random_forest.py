#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:14:45 2024

apply random forest and see how the results look like.

@author: fangyichen
"""
#TODO Delete spaghetti Codes.
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

# %%
# Import x_train_tfidf, x_test_tfidf, y_train, y_test, x_train_df, features.

import text_tfidf_vectorizing
x_train_tfidf, x_test_tfidf, y_train, y_test, x_train_df, x_test_df, features, y = text_tfidf_vectorizing.text_tfidf_vectorizing()

x = vstack((x_train_tfidf, x_test_tfidf)) 

# %%
#apply random forest.
rf = RandomForestClassifier()
rf = rf.fit(x_train_tfidf, y_train)

#Do prediction.
y_pred = rf.predict(x_test_tfidf)


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
#Evaluate the results.

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of standard model:", accuracy)
#the model predicted correct in 87.7% of time.

# %%
#View confusion matrix for test data and predictions
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
#73 Actual Positive got predicted as negative
#47 actual negative got predicted as positive

# %%
## View the classification report for test data and predictions
print(classification_report(y_test, y_pred))
#Precision: The ratio of true positives to the sum of true positives and false positives. 
#0 is human written text and 1 ml text.
#When model predict it is human written 85% is true.
#When model predict it is ml written 90% is true.

#Recall:  The ratio of true positives to the sum of true positives and false negatives. 
#The model can predict 90% of human written text as human written and miss 10%.
#The model can predict 85% of ml text as ml and miss 10%.

#F1 Score: The harmonic mean of precision and recall. 

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
# Find put the features that contribute to 95% percent of importances.
# List of features sorted from most to least important
sorted_importances = list(feat_importances_sorted.values)
sorted_features = list(feat_importances_sorted.index)
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
plt.plot(features, cumulative_importances, 'g-')
# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(features, sorted_features, rotation = 'vertical')
# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');

# %%
# Find number of features for cumulative importance of 95%
# Add 1 because Python is zero-indexed

feature_number = np.where(cumulative_importances > 0.95)[0][0] + 1
print('Number of features for 95% importance:', feature_number)

# %%
#TODO
# Extract the names of the most important features and filter the training and testing dataset x.
important_feature = sorted_features[0:feature_number]

# Create training and testing sets with only the important features
'''
y_important_feature = tdidf_important_feature['label']

x_important_feature = tdidf_important_feature.drop("label", axis=1, inplace=False)

x_important_feature = x_important_feature.values
'''
#Split the data into training and testing data sets.
x_train_if, x_test_if, y_train_if, y_test_if = train_test_split(x_important_feature, y_important_feature, test_size = 0.2,random_state = 4)

# %%
rf_if = RandomForestClassifier()
rf_if = rf_if.fit(x_train_if, y_train_if)

#Do prediction.
y_pred_if = rf_if.predict(x_test_if)

# %%
#Evaluate the results.
accuracy_rf_if = accuracy_score(y_test_if, y_pred_if)
print("Accuracy of model applied random grid:", accuracy_rf_if)

# %%
#View confusion matrix for test data and predictions
print("Confusion matrix:", confusion_matrix(y_test_if, y_pred_if))

# %%
## View the classification report for test data and predictions
print(classification_report(y_test_if,y_pred_if))

'''
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
rf_if = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_if_random = RandomizedSearchCV(estimator = rf_if, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_if_random.fit(x_train_if, y_train_if)

print(rf_if_random.best_params_)

#Result: {'n_estimators': 1400, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': True}
'''

# %%
#Do prediction.

from sklearn.ensemble import RandomForestClassifier

rf_if_hp = RandomForestClassifier(
    n_estimators=1400,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features="sqrt",
    max_depth=70,
    bootstrap=True
)

rf_if_hp = rf_if_hp.fit(x_train_if, y_train_if)
y_pred_if_hp = rf_if_hp.predict(x_test_if)

# %%
#Evaluate the results.
accuracy_rf_hp = accuracy_score(y_test_if, y_pred_if_hp)
print("Accuracy of model applied random grid and hyperparameter optimization:", accuracy_rf_hp)

# %%
#View confusion matrix for test data and predictions
print("Confusion matrix:", confusion_matrix(y_test_if, y_pred_if_hp))

# %%
## View the classification report for test data and predictions
print(classification_report(y_test_if,y_pred_if_hp))
