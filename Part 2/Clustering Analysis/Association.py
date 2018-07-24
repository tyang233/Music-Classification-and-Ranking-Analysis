#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 23:32:55 2017

@author: chengzhong
"""


#set the packages needed
import pandas as pd 
from apyori import apriori
import sys

def association(argv):
    aprioridf = myapriori()
    print(aprioridf)

#define the apriori function
def myapriori():
    #set three levels of minimum support and confidence
    minsup_con = [(0.8,0.8),(0.75,0.75),(0.7,0.7)]
    #run the apriori algorithm
    for i,j in minsup_con:
        #read the data
        spotifydf = pd.read_csv("mySpotify_v3.csv")
        #drop the duplicate values
        spotifydf = spotifydf.drop_duplicates(subset = 'track_id',keep='first',inplace = False)
        #run apriori
        itemset = spotifydf[['parentCat','acousticness_bin','danceability_bin','energy_bin',
                      'instrumentalness_bin','liveness_bin','loudness_bin','speechiness_bin',
                      'tempo_bin','time_signature_bin','valence_bin']].values.tolist()
        #set the result as list
        results = list(apriori(itemset,min_support = i, min_confidence = j))
        aprioridf = pd.DataFrame(results)
    return(aprioridf)

