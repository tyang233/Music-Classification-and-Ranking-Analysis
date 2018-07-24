# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:34:15 2017

@author: minxiaocn
"""


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


 ######################################################
# For the best model, see how well it does on the
# validation test
######################################################
def RF(X_train,Y_train,X_validate,Y_validate):

    clf=RandomForestClassifier(max_depth=2,random_state=0)
    
    ## ==== Evaluation the model ===
    num_folds=10
    
    seed = 7
    scoring = 'accuracy'
    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
    cv_results = cross_val_score(clf, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ("Random Forest training", cv_results.mean(), cv_results.std())
    print(msg)
    
     ######################################################
     # For the best model (RF), see how well it does on the
     # validation test
     ######################################################
    # Make predictions on validation dataset
    clf=RandomForestClassifier(max_depth=2,random_state=0)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_validate)
    
    print("classification model using random forest")
    print(accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))
    
    with open("classification.txt","a") as f:
        f.write(msg+"\n")
        f.write("confusion matrix \n"+str(confusion_matrix(Y_validate, predictions))+"\n")
        f.write(str(classification_report(Y_validate, predictions)))
        f.write("\n")  
    
     ## ROC for classifier on validation data
    #compute fpr, tpr, thresholds and roc auc
    y_score = clf.fit(X_train, Y_train).predict_proba(X_validate)
    return y_score[:,1],cv_results.mean(), accuracy_score(Y_validate, predictions)
    
    
    
    
   




