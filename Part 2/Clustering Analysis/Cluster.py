#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:38:24 2017

@author: chengzhong
"""
#Import all the packages we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import preprocessing
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from pprint import pprint
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial.distance import cdist
import sys

#define the main funtion for clustering
def cluster(argv): 
    #read the data as a dataframe
    mySpotify = pd.read_csv("myspotify_v3.csv",sep=',', encoding='utf-8')
    #Choose the pop set from the whole dataset
    myData = mySpotify[mySpotify['parentCat']=='pop']
    #drop all the duplicate values
    myData = myData.drop_duplicates(subset = 'track_id',keep='first',inplace = False) 
    myData1=pd.concat([myData['duration_ms'], myData['acousticness'], myData['danceability'], myData['energy'], myData['key'], 
                      myData['liveness'], myData['loudness']], 
                     axis=1, keys=['duration_ms', 'acousticness', 'danceability', 'energy', 'key','liveness', 'loudness' ])
    #returns a numpy array
    x = myData1.values 
    #normalize the data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    #Perform all the clustering
    mykmeans(normalizedDataFrame,x_scaled,myData1)
    myhierach(x_scaled)
    mydbscan(x_scaled)

# define teh k means cluster function
def mykmeans(normalizedDataFrame,x_scaled,myData1):
    # Set the normalized data as X
    X = x_scaled
    # Set the estimators for different k
    estimators = [('k_means_2',2, KMeans(n_clusters=2)),
                  ('k_means_3',3, KMeans(n_clusters=3)),
                  ('k_means_5',5, KMeans(n_clusters=5)),
                  ('k_means_20',20, KMeans(n_clusters=20))]
    cluster = [2,3,5,20]
    # Set a list for mean distance between clusters
    meandist = []
    #plot the clusters
    for name, k,est in estimators:
        kmeans = KMeans(n_clusters=k)
        #set the cluster label
        cluster_labels = kmeans.fit_predict(normalizedDataFrame)
        #calculate the centroids of the cluster
        centroids = kmeans.cluster_centers_
        #print the centroids
        #pprint(centroids)
        #calculate the mean distance between clusters
        meandist.append(sum(np.min(cdist(x_scaled,kmeans.cluster_centers_,'euclidean'),axis=1))/x_scaled.shape[0])
        #set and print the silhouette coefficient
        silhouette_avg = silhouette_score(x_scaled,cluster_labels)
        print("For kmeans n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
        #plot the 3d cluster plot
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X)
        labels = est.labels_
        ax.scatter(X[:, 2], X[:, 1], X[:, 3],
                   c=labels.astype(np.float), edgecolor='k')
        #set plot labels
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('danceability')
        ax.set_ylabel('acousticness')
        ax.set_zlabel('energy')
        ax.set_title(name)
        ax.dist = 12
    plt.figure()
    #plot the mean distance vs k
    plt.plot(cluster,meandist)
    plt.xlabel("Number of clusters")
    plt.ylabel("Mean distance")
    plt.title("Mean distance vs K")
    
#define the hierarchical function
def myhierach(x_scaled):
    #set the for loop for different number of clusters
    for k in (2,3,5,20):
        #Set the normalized data
        X=x_scaled
        #calculate the hierarchical cluster
        ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X)
        #set the cluster labels
        label = ward.labels_
        #plot the cluster
        fig = plt.figure()
        ax = p3.Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.view_init(7, -80)
        for l in np.unique(label):
            ax.scatter(X[label == l, 1], X[label == l, 2], X[label == l, 3],
                       color=plt.cm.jet(np.float(l) / np.max(label + 1)),
                       s=20, edgecolor='k')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('acousticness')
        ax.set_ylabel('danceability')
        ax.set_zlabel('energy')
        ax.set_title("Hierarchical Clustering n= %d"%k)
        ax.dist = 12
        #print the silhouette coefficient
        print("For Hierarchical Clustering n=", k ,"the silhouette coefficient is: %0.3f"
          % metrics.silhouette_score(X, label))
        plt.show()

#define the dbscan function
def mydbscan(x_scaled):
    X=x_scaled
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=20).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("For DBSCAN the silhouette coefficient is: %0.3f"
          % metrics.silhouette_score(X, labels))
    # Plot result
    # Black removed and is used for noise instead.
    plt.figure()
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    #set the title of the plot
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
