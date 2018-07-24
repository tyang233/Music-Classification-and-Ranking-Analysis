#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:27:43 2017
Basic Statistical Aanalysis and data cleaning insight
Histograms and Correlations
@author: Tianyu Yang
"""
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def desc_stats(unique_track,track):
    #Descrptive Statistics
    varz = ['acousticness','danceability', 'energy','instrumentalness', 'liveness','loudness','mode',
            'speechiness','tempo','time_signature','valence']
    
    #show means, median, standard deviation of the above variables
    unique_track[varz].describe().loc[['mean','50%','std']].to_csv('desc_stats.csv')
    print(unique_track[varz].describe().loc[['mean','50%','std']])
    #show modes for categories
    print('modes for categories')
    for name,group in track.groupby('parentCat').groups.items():
        print(name, len(group))
def id_outliers(track):
    
    quant_varz = ['acousticness','danceability', 'energy','instrumentalness', 'liveness','loudness',
            'speechiness','tempo','time_signature','valence']
    
    #identify general outliers and variable specific outliers in the above variables using interquartile range
    #a observation is a general outlier if at least one of the quantitative variables is labeled as outlier
    #0 is false and 1 is true
    unique_track = track.drop_duplicates('track_id')
    track['general_out']=0
    
    
    #select unique track 
    for var in quant_varz:
        
        #using IQR and quantiles calculated using unique tracks    
        q1 = unique_track[var].quantile(0.25)
        q3 = unique_track[var].quantile(0.75)
        iqr = q3-q1
        track[var+'_out'] = 0
        track.loc[(track[var] < (q1-1.5*iqr)) | (track[var]>(q3+1.5*iqr)),[var+'_out','general_out']] = 1
    return track

def boxplots(unique_track):
    #show outliers using boxplot
    #only put the ones with min 0 and max 1 on the same graph
    plt.figure()
    unique_track[['acousticness', 'danceability', 'energy']].boxplot()
    plt.figure()
    unique_track[['instrumentalness', 'liveness', 'speechiness', 'valence']].boxplot()
    plt.figure()
    unique_track['loudness'].plot(kind = 'box')
    plt.figure()
    unique_track['tempo'].plot(kind ='box')
    plt.figure()
    unique_track['time_signature'].plot(kind = 'box')
    
    
def binning(track):  
    #bin the quantitative variables
    quant_varz = ['acousticness','danceability', 'energy','instrumentalness', 'liveness','loudness',
            'speechiness','tempo','time_signature','valence']
    for var in quant_varz:
        track[var+'_bin'] = pd.cut(track[var],3,labels = ['low'+var,'medium'+var,'high'+var])
    
    return track

def hist_scatter(unique_track):
    #histograms and scatterplots
    vars_of_interest = ['acousticness','danceability', 'energy','instrumentalness', 'liveness','loudness',
            'speechiness','tempo','time_signature','valence']
    for var in vars_of_interest:
        plt.figure()
        #the histogram shows counts,
        plt.hist(unique_track[var])
        plt.ylabel('Counts')
        plt.xlabel(var)
        plt.title('Histogram of '+var)
    
    #correlation matrix and interpretation using pairwise scatter plot
    plt.figure()
    scatter_matrix(unique_track[vars_of_interest])
    unique_track[vars_of_interest].corr().to_csv('correlation.csv')
    print('\nCorrelation Matrix')
    print(unique_track[vars_of_interest].corr())
    
def main():
    
    #Read in spotify data
    tracks = pd.read_csv('mySpotify_v2.csv')
    unique = tracks.drop_duplicates('track_id')
    boxplots(unique)
    desc_stats(unique,tracks)
    tracks = binning(id_outliers(tracks))
    hist_scatter(unique)
    tracks.to_csv('mySpotify_v3.csv')

main()
