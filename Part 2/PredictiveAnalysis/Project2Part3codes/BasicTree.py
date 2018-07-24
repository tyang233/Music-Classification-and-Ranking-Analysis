# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:46:34 2017

@author: minxiaocn
"""

### ### ### ### #####
## == Decision Tree #
## ### ### ### ### ##
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

def Basic_DecisionTree(X_train,Y_train,X_validate,Y_validate,genre_target):
    clf=tree.DecisionTreeClassifier(max_depth=4)
    #build a tree and get a graph
    clf2=clf.fit(X_train,Y_train)
    feature_nms=['duration_ms' ,'popularity','acousticness',
    'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness',
    'mode','speechiness','tempo','time_signature','valence']
    dot_data=tree.export_graphviz(clf2,out_file=genre_target+"graph.gv",feature_names=feature_nms,
                                  class_names=["0","1"],filled=True,rounded=True,special_characters=True)
    graph=graphviz.Source(dot_data)
    graph
    
 
    
    ## ==== Evaluation the model ===
    num_folds=10
    seed = 7
    scoring = 'accuracy'
    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
    cv_results = cross_val_score(clf, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ("decisionTree training", cv_results.mean(), cv_results.std())
    print(msg)
     ######################################################
     # For the best model (CART), see how well it does on the
     # validation test
     ######################################################
     # Make predictions on validation dataset
    print("classification model using decison tree")
    
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_validate)
   
    
    print(accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))
    with open("classification.txt","a") as f:
        f.write(msg+"\n")
        f.write("confusion matrix \n"+str(confusion_matrix(Y_validate, predictions)))
        f.write(str(classification_report(Y_validate, predictions)))
        f.write("\n")
        
    
    
     ## ROC for classifier on validation data
    #compute fpr, tpr, thresholds and roc auc
    y_score = clf.fit(X_train, Y_train).predict_proba(X_validate)
    
    return y_score[:,1],cv_results.mean(), accuracy_score(Y_validate, predictions)
    
    
    
    
 
    
