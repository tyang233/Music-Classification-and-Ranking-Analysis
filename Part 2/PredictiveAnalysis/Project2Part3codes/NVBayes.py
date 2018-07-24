# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:33:34 2017

@author: minxiaocn
"""

#KNN
# Import all your libraries.
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



#os.chdir



 ######################################################
# For the best model, see how well it does on the
# validation test
######################################################
def NVBayes(X_train,Y_train,X_validate,Y_validate):
    # Make predictions on validation dataset
    gnb = GaussianNB()
    
    ## ==== Evaluation the model ===
    num_folds=10
   
    seed = 7
    scoring = 'accuracy'
    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
    cv_results = cross_val_score(gnb, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ("Naive Bayes training", cv_results.mean(), cv_results.std())
    print(msg)
    
     ######################################################
     # For the best model (CART), see how well it does on the
     # validation test
     ######################################################
    # Make predictions on validation dataset
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    predictions = gnb.predict(X_validate)
    print(predictions)
    
    print("classification model using naive bayes")
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
     ## ROC for classifier on validation data
    #compute fpr, tpr, thresholds and roc auc
    y_score = gnb.fit(X_train, Y_train).predict_proba(X_validate)
    return y_score[:,1],cv_results.mean(), accuracy_score(Y_validate, predictions)
    
    
 
    
    
    
   

   