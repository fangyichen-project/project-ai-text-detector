#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementing support vector machine.

@author: fangyichen
"""

from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import copy as cp
from typing import Tuple
from scipy.sparse import vstack

# %%
# Import x_train_tfidf, x_test_tfidf, y_train, y_test, x_train_df, features.

import text_tfidf_vectorizing

x_train_tfidf, x_test_tfidf, y_train, y_test, x_train_df, features, x, y = text_tfidf_vectorizing.text_tfidf_vectorizing()


# %%
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(x_train_tfidf,y_train)
# predict the labels on validation dataset
y_pred = SVM.predict(x_test_tfidf)
# Use accuracy_score function to get the accuracy

print("SVM Accuracy Score -> ",accuracy_score(y_pred,y_test)*100)

# %%
## View the classification report for test data and predictions

print(classification_report(y_test,y_pred))

# %%
'''
#hyper tune your mode

grid = {
    'C':[0.01,0.1,1,10],
    'kernel' : ["linear","poly","rbf","sigmoid"],
    'degree' : [1,3,5,7],
    'gamma' : [0.01,1]
}

svm  = SVC()
svm_cv = GridSearchCV(svm, grid, cv = 5)
svm_cv.fit(X_train,y_train)

print("Best Parameters:",svm_cv.best_params_)

print("Train Score:",svm_cv.best_score_)

print("Test Score:",svm_cv.score(X_test,y_test))
'''
#Best Parameters: {'C': 10, 'degree': 1, 'gamma': 1, 'kernel': 'rbf'}
#Train Score: 0.925002225189141
#Test Score: 0.9263607257203842

# %%
# fit the dataset to training data.
SVM_tuned = svm.SVC(C=10, degree=1, gamma=1, kernel='rbf')
SVM_tuned.fit(x_train_tfidf,y_train)
# predict the labels on validation dataset
y_pred_tuned = SVM_tuned.predict(x_test_tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(y_pred_tuned,y_test)*100)

# %%
# K-Fold Cross-Validation.
kfold = KFold(n_splits=5, random_state=4, shuffle=True)
cv_results = cross_val_score(SVM_tuned, x_train_tfidf, y_train, cv=kfold, scoring='accuracy', verbose=10)

print("results of k-fold cross-validation:", cv_results.mean(), cv_results.std())

# %%
# Plot a Confusion Matrix from a K-Fold Cross-Validation.
def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:

    model_ = cp.deepcopy(model)
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba

# %%
# Visualise the Confusion Matrix.

# Calculate the confusion matrix
actual_classes, predicted_classes, predicted_proba = cross_val_predict(SVM_tuned, kfold, x.toarray(), y.values)

cm = confusion_matrix(actual_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#TODO label the matrix better.
# %%
from sklearn.metrics import balanced_accuracy_score

# Calculate Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)

#%%
#Define the output.
def support_vector_machine():
    print(f'Balanced Accuracy: {balanced_acc}')
    print("results of k-fold cross-validation:", cv_results.mean(), cv_results.std())
    print("SVM Accuracy Score:", accuracy_score(y_pred_tuned,y_test)*100)
    print("Classification report svm:", classification_report(y_test,y_pred))
    plt.show()
    
if __name__ == "__main__":
    support_vector_machine()
    



















