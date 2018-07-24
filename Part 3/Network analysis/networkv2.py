#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:15:17 2017

@author: tytys
"""
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def similarity(artists, thresh = 0.5):
    '''return a list of lists of binary code for whether the artists are similar'''
    # measure the euclidean distanace of two artists with given attributes
    # two artists are similar if the euclidean distance is below a given threshold
    return [artists.apply(lambda x: int(np.sum(np.sqrt((x-artists.loc[artist])**2)) <= thresh), axis = 1).tolist() for artist in artists.index ]


def weighted_edges(artists,from_artists, to_artists):
    '''return a weighted edge list'''
    
    weighted_edges = []
    
    #put each pair of edge with weight into a tuple 
    for k in range(len(from_artists)):
        for i,j in zip(from_artists[k], to_artists[k]):
            
            #do not include self connection
            if i!=j:
                
                #weight is determined by e^(-x)*10 where x is the euclidean distance
                #Therefore, weight will be define on all real numbers and have a maximum of 10
                weighted_edges.append((i,j,np.exp(-np.sum(np.sqrt((artists.loc[i] - artists.loc[j])**2)))*10))
    return weighted_edges

def labels(G, threshhold = 95):
    '''return labels(dictionary) for nodes with high centrality for a given percentile'''
    labels = {}
    
    # create cutoff based on the given percentile
    cen_cutoff = np.percentile(list(nx.degree_centrality(G).values()), threshhold)
    
    # put nodes label in the dictionary if the centrality passes the threshold
    for key,value in nx.degree_centrality(G).items():
        if value >= cen_cutoff:
            labels[key] = key
    
    return labels

def show_network_metrics(G):
    '''
    Print the local and global metrics of the network
    '''
    print(nx.info(G))

    # density
    print("Density of the network")
    print(nx.density(G))    
    
    # average  betweeness
    print("Average  betweeness of the network")
    print(np.sum(list(nx.betweenness_centrality(G).values()))/len(nx.betweenness_centrality(G)))

    # Average clustering coefficient
    print("Average clustering coefficient:")
    print(nx.average_clustering(G))


    #create metrics dataframe
    by_node_metrics = pd.DataFrame({"Betweeness_Centrality":nx.betweenness_centrality(G),"Degree_Centrality":nx.degree_centrality(G),
        "Clustering_Coefficient":nx.clustering(G), "Triangels":nx.algorithms.cluster.triangles(G)})
    print(by_node_metrics)

    by_node_metrics.to_excel("metrics.xlsx")

def draw_network(tracks, cat = None, num = 100):
    
    unique = tracks.drop_duplicates('track_id')
    if cat:
        unique = unique[unique['parentCat']==cat]
    
    #only select artists with high popularity
    artists = unique.groupby('artists')[['popularity','acousticness','danceability','energy',
                             'instrumentalness','speechiness','valence']].mean().sort_values(by = 'popularity', ascending  = False)[:num]
    
    # calculate similarity
    simi = similarity(artists, thresh=1)
    
    # translate binary code in similarity to artists
    to_artists_i = [[i for i in range(len(simi[j])) if simi[j][i] ==1] for j in range(len(simi))]
    to_artists = [[artists.index[i] for i in to_artists_i[j]] for j in range(len(to_artists_i))]
    
    # pair the to artists with from_artists based on the outer index of the to_artists list
    from_artists = [[artists.index[j] for i in range(len(to_artists[j]))] for j in range(len(to_artists))]
 
    #create edges among artists
    w_edges = weighted_edges(artists,from_artists, to_artists)

    #create network graph
    g = nx.Graph()
    
    #add nodes and edges
    g.add_nodes_from(artists.index)
    g.add_weighted_edges_from(w_edges)
    
    #add creat labels for nodes with high centrality
    lab = labels(g, threshhold = 90)
    
    #draw the graph 
    plt.figure()
    nx.draw_spring(g, node_color =list(nx.degree_centrality(g).values()),
        node_size=100, with_labels=True,labels =lab, font_size  =10, font_color = 'r')
    plt.show()
    
    #print metircs
    show_network_metrics(g)

def main():
    tracks = pd.read_csv('mySpotify_v3.csv', index_col =0)
    draw_network(tracks)
    


main()



