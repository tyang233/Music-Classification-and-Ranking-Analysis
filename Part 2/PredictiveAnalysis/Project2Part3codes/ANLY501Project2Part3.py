# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:58:42 2017

@author: minxiaocn
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from BasicTree import *
from KNN import *
from NVBayes import *
from SVM import *
from RF import *
#os.chdir

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy import stats


    
def main():

    #prepare for data
    rawdata=pd.read_csv("mySpotify.csv")
    
    
    
    newdata=rawdata[['track_id','duration_ms' ,'popularity','acousticness',
     'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness',
     'mode','speechiness','tempo','time_signature','valence','parentCat']]
    
    #96153
    
    ## hypothesis 1: t test : mean  of energy between pop and non-pop
    print("hypothesis 1")
    target_pop=newdata[newdata.parentCat=="pop"].drop_duplicates('track_id')
    id_in_labels=target_pop['track_id'].values
    non_pop=newdata[~newdata.track_id.isin(id_in_labels)].drop_duplicates('track_id')
    
    print(stats.ttest_ind(target_pop["energy"].values,non_pop["energy"].values))
    
    #hypothesis 2: linear regression acousticness~energy+loudness
    print("hypothesis 2")
    data_unique=newdata.drop_duplicates('track_id')
    X = data_unique[["energy",'loudness']]
    X = sm.add_constant(X)
    y = data_unique["acousticness"]
    
    # Note the difference in argument order
    model = sm.OLS(y, X).fit() 
    # Print out the statistics
    print(model.summary())
    
    ## == groupby data using gernre ==
    #there are 29 genres
    print("hypothesis 3")
    genres=newdata["parentCat"].unique()
    with open("genres_spotify.txt","w") as f:
        f.write(str(genres))
    
    print(genres)
    #record the classification result of diffrent classification methods on all genres
    precision_train=pd.DataFrame(0,index=genres,columns=["DecisisionTree","KNN","Naive Bayes","SVM","Random Forest"])
    precision_test=pd.DataFrame(0,index=genres,columns=["DecisisionTree","KNN","Naive Bayes","SVM","Random Forest"])
    
    with open("classification.txt","w") as f:
        f.write("classification result \n")
        
       
    for i in np.arange(len(genres)):
        with open("classification.txt","a") as f:
            f.write("\n classification result for "+ genres[i]+"\n")
        #state the variable using string
        vars()["data_"+genres[i]]=newdata[newdata.parentCat==genres[i]]
    
        
        data_target=vars()["data_"+genres[i]]
        genre_target=genres[i]
        if genres[i] in ["party","workout"]:
            data_target=data_target.sample(3000)
            
        ##fill with col index
        #features_selection=[]
        #y_selection=15
        
        #Fisrt use data_Classical as the target data, then we randomly sample song tracks which dosen't 
        #belong to Classical and integrate them into the data set
        # 
        id_in_labels=data_target['track_id'].values
        data_out_of_label=newdata[~newdata.track_id.isin(id_in_labels)].drop_duplicates('track_id').sample(n=len(data_target))
        #normalize dataframe columns
        
        #print(data_target.describe())
        #print(data_out_of_label.describe())
        
        #combine two data sets into one
        df=pd.concat([data_out_of_label,data_target]).sort_values(by=["track_id"])
        df.parentCat[df.parentCat !=genre_target]=0
        df.parentCat[df.parentCat==genre_target]=1
        
        
        
        
        ##normalize data
        from sklearn import preprocessing
        x=df.iloc[:,1:15] #the last col is not included
        min_max_scaler=preprocessing.MinMaxScaler()
        x_scaled=min_max_scaler.fit_transform(x)
        df.iloc[:,1:15]=x_scaled
        
        
        
        ##set training and test datasets;
        df_values=df.values
        X=np.array(df_values[:,1:15].tolist())
        Y=np.array(df_values[:,15].tolist())
        test_size = 0.30
        seed = np.random.randint(1,4)
        X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
        
        
        
        
        print("classification results for "+genres[i])
        
        #### ===== ======#
        # Decison Tree ###
        ##################
        
        y_score,train_precision,test_precision=Basic_DecisionTree(X_train,Y_train,X_validate,Y_validate,genre_target)
        fpr_1, tpr_1, thresholds=roc_curve(Y_validate,y_score)
        roc_auc_1=auc(fpr_1, tpr_1)
        precision_train.loc[genres[i],"DecisisionTree"]=train_precision
        precision_test.loc[genres[i],"DecisisionTree"]=test_precision
    
                  
         #plot roc curve
    #    fig = plt.figure()
    #    plt.plot(fpr,tpr,label="ROC curve(area =%0.2f)" %roc_auc)
    #    plt.plot([0,1],[0,1],"k--")
    #    plt.xlim([0.0,1.0])
    #    plt.ylim([0.0,1.0])
    #    plt.xlabel("False positive ")
    #    plt.ylabel("true positive")
    #    plt.title("ROC of Decision Tree classifier for"+genre_target+",area=="+str(roc_auc))
    #    
    #    fig.savefig('ROC DecisionTree'+genre_target+'.png')
    #    plt.close()
         
        
        ####### ===== ======#
        #      KNN       ###
        #####################
        y_score,train_precision,test_precision=KNN(X_train,Y_train,X_validate,Y_validate)
        fpr_2, tpr_2, thresholds=roc_curve(Y_validate,y_score)
        roc_auc_2=auc(fpr_2, tpr_2)
        
        precision_train.loc[genres[i],"KNN"]=train_precision
        precision_test.loc[genres[i],"KNN"]=test_precision
        
       
        #### ===== ======#
        # Naive bayes ###
        ##################
        y_score,train_precision,test_precision=NVBayes(X_train,Y_train,X_validate,Y_validate)
        fpr_3, tpr_3, thresholds=roc_curve(Y_validate,y_score)
        roc_auc_3=auc(fpr_3, tpr_3)
        
        precision_train.loc[genres[i],"Naive Bayes"]=train_precision
        precision_test.loc[genres[i],"Naive Bayes"]=test_precision
        
    #     
        #### ===== ======#
        # Svm ###
        ##################
        y_score,train_precision,test_precision=mysvm(X_train,Y_train,X_validate,Y_validate)
        fpr_4, tpr_4, thresholds=roc_curve(Y_validate,y_score)
        roc_auc_4=auc(fpr_4, tpr_4)
        
        precision_train.loc[genres[i],"SVM"]=train_precision
        precision_test.loc[genres[i],"SVM"]=test_precision
        
       #### ===== ======#
        # RandomForest ###
        ##################
        y_score,train_precision,test_precision=RF(X_train,Y_train,X_validate,Y_validate)
        fpr_5, tpr_5, thresholds=roc_curve(Y_validate,y_score)
        roc_auc_5=auc(fpr_5, tpr_5)
        
        precision_train.loc[genres[i],"Random Forest"]=train_precision
        precision_test.loc[genres[i],"Random Forest"]=test_precision
        
         #plot roc curve
        fig = plt.figure()
        plt.plot(fpr_1,tpr_1,label="ROC curve for decisionTree(area =%0.2f)" %roc_auc_1,color='darkorange')
        plt.plot(fpr_2,tpr_2,label="ROC curve for KNN(area =%0.2f)" %roc_auc_2,color='aqua')
        plt.plot(fpr_3,tpr_3,label="ROC curve for Naive bayes(area =%0.2f)" %roc_auc_3,color='cornflowerblue')
        plt.plot(fpr_4,tpr_4,label="ROC curve for SVM (area =%0.2f)" %roc_auc_4,color='green')
        plt.plot(fpr_5,tpr_5,label="ROC curve for RandomForest(area =%0.2f)" %roc_auc_5,color='red')
        
        plt.plot([0,1],[0,1],"k--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel("False positive ")
        plt.ylabel("true positive")
        plt.title("ROC of classifiers for"+genre_target)
        plt.legend(loc="lower right")
        fig.savefig('ROC '+genre_target+'.png')
        plt.close()
    
    precision_train.to_csv("precision_train.csv")
    precision_test.to_csv("precision_test.csv") 
        
        #ROC : https://kennis-research.shinyapps.io/ROC-Curves/
    
if __name__== "__main__":
     main()
    
    
 