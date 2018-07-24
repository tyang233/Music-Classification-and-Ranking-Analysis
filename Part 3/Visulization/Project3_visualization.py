#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:06:57 2017

@author: chengzhong
"""

import plotly
import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np


def main():
    #Set up credentials
    plotly.tools.set_credentials_file(username='bryanzhongcheng', 
                                      api_key='G7tVHiOxGYppWFoJqELk')
    # Setup dataframe
    dataFrame = pd.read_csv('mySpotify_v3.csv')
    billboard = pd.read_csv('myBillboard.csv')
    pop = dataFrame[dataFrame['parentCat']=='pop']
    #Run all the visualization functions
    top_five_camparision(dataFrame)
    scatter_bubble_vis(dataFrame,pop)
    ranking_trend(billboard)
    title_wordcloud(dataFrame)


#define the trend comparing function
def top_five_camparision(dataFrame):
    #get the top song information from dataFrame 17
    one_17 = dataFrame[dataFrame['track_name']=='Attention'].iloc[[0]]
    two_17 = dataFrame[dataFrame['track_name']=='Slow Hands'].iloc[[0]]
    three_17 = dataFrame[dataFrame['track_name']=='Strip That Down'].iloc[[0]]
    four_17 = dataFrame[dataFrame['track_name']=="There's Nothing Holdin' Me Back"].iloc[[0]]
    five_17 = dataFrame[dataFrame['track_name']=="Believer"].iloc[[0]]
    #concat 17 information
    ranking_17 = pd.concat([one_17,two_17,three_17,four_17,five_17])[['artists',
                          'track_name','acousticness','danceability','energy','valence']]
    #get the top song information from dataFrame 16                  
    one_16 = dataFrame[dataFrame['track_name']=='Closer'][dataFrame['artists']=='The Chainsmokers'].iloc[[0]]
    two_16 = dataFrame[dataFrame['track_name']=='Cold Water'].iloc[[2]]
    three_16 = dataFrame[dataFrame['track_name']=='Treat You Better'].iloc[[0]]
    four_16 = dataFrame[dataFrame['track_name']=="Cheap Thrills"].iloc[[0]]
    five_16 = dataFrame[dataFrame['track_name']=="Send My Love (To Your New Lover)"].iloc[[0]]
    #concat 16 information
    ranking_16 = pd.concat([one_16,two_16,three_16,four_16,five_16])[['artists',
                          'track_name','acousticness','danceability','energy','valence']]
    
    #get the linemode visualization
    data1_17= go.Scatter(
        x = ranking_17['track_name'].tolist(),
        y = ranking_17['acousticness'].tolist(),
        mode = 'lines+markers',
        name ='acousticness17',
        marker = dict(color = 'blue')
    )
    
    data1_16= go.Scatter(
        x = ranking_16['track_name'].tolist(),
        y = ranking_16['acousticness'].tolist(),
        mode = 'lines+markers',
        name ='acousticness16',
        marker = dict(color = 'blue')
    )
    
    data2_17= go.Scatter(
        x = ranking_17['track_name'].tolist(),
        y = ranking_17['danceability'].tolist(),
        mode = 'lines+markers',
        name ='danceability17',
        marker = dict(color = 'red')
    )
    
    data2_16= go.Scatter(
        x = ranking_16['track_name'].tolist(),
        y = ranking_16['danceability'].tolist(),
        mode = 'lines+markers',
        name ='danceability16',
        marker = dict(color = 'red')
    )
    
    data3_17= go.Scatter(
        x = ranking_17['track_name'].tolist(),
        y = ranking_17['energy'].tolist(),
        mode = 'lines+markers',
        name ='energy17',
        marker = dict(color = 'green')
    )
    
    data3_16= go.Scatter(
        x = ranking_16['track_name'].tolist(),
        y = ranking_16['energy'].tolist(),
        mode = 'lines+markers',
        name ='energy16',
        marker = dict(color = 'green')
    )
    
    data4_17= go.Scatter(
        x = ranking_17['track_name'].tolist(),
        y = ranking_17['valence'].tolist(),
        mode = 'lines+markers',
        name ='valence17',
        marker = dict(color = 'orange')
    )
    
    data4_16= go.Scatter(
        x = ranking_16['track_name'].tolist(),
        y = ranking_16['valence'].tolist(),
        mode = 'lines+markers',
        name ='valence16',
        marker = dict(color = 'orange')
    )
    #set the legends
    data = [data1_17,data1_16,data2_17,data2_16,data3_17,data3_16,data4_16,data4_17]
    layout = go.Layout(
        title='Sound Feature Comparision From 2016 to 2017',
        yaxis=dict(title='Sound Feature')
    )
    figure = go.Figure(data=data, layout=layout)
    plot(figure, filename='ranking_17.html')

#define the function creating scatter bubble plot
def scatter_bubble_vis(dataFrame,pop):
    #create the bubble chart
    data = [
        {
            'x': pop['acousticness'],
            'y': pop['energy'],
            'mode': 'markers',
            'marker': {
                'color':pop['time_signature'] ,
                'size': pop['key'],
                'showscale': True
            }
        }
    ]
    #set the legend
    layout = go.Layout(
        title='Acousticsness Vs Energy',
        xaxis=dict(title='acousticness'),
        yaxis=dict(title='energy')
    )
    fig = go.Figure(data=data, layout=layout)
    
    #plot
    plot(fig, filename='Acousticsness Vs Energy.html')
    
    #create the bubble chart
    data = [
        {
            'x': pop['danceability'],
            'y': pop['acousticness'],
            'mode': 'markers',
            'marker': {
                'color':pop['time_signature'] ,
                'size': pop['key'],
                'showscale': True
            }
        }
    ]
    #set the legend
    layout = go.Layout(
        title='Danceability Vs Acousticsness',
        xaxis=dict(title='Danceability'),
        yaxis=dict(title='Acousticness')
    )
    fig1 = go.Figure(data=data, layout=layout)
    #plot
    plot(fig1, filename='Danceability Vs Acousticsness.html')
    
    #create the bubble chart
    data = [
        {
            'x': pop['danceability'],
            'y': pop['energy'],
            'mode': 'markers',
            'marker': {
                'color':pop['time_signature'] ,
                'size': pop['key'],
                'showscale': True
            }
        }
    ]
    
    layout = go.Layout(
        title='Danceability Vs Energy',
        xaxis=dict(title='Danceability'),
        yaxis=dict(title='Energy')
    )
    fig2 = go.Figure(data=data, layout=layout)
    #plot
    plot(fig2, filename='Danceability Vs Energy.html')

#define the function to plot the ranking trend
def ranking_trend(billboard):
    #set the dataframe
    df2 = [rows for _, rows in billboard.groupby('Genre')]
    df3 = [rows for _, rows in df2[7].groupby('artist')]
    artist_name = ['Drake','Adele','Rihanna','Lady Gaga','Kanye West','Calvin Harris']
    data = [0]*len(artist_name)
    count = 0
    
    for i in range(len(df3)):
        if list(df3[i]['artist'])[0] in artist_name:
            # Create traces
            data[count]= go.Scatter(
                x = df3[i]['Date'],
                y = df3[i]['rank'],
                mode = 'lines+markers',
                name = list(df3[i]['artist'])[0]
            )
            count = count+1
    layout = go.Layout(
        title='Billboard Ranking Trend',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Rank')
    )
    data1 = go.Figure(data=data, layout=layout)
    plot(data1, filename='line-mode.html')

#define the function to plot word cloud
def title_wordcloud(dataFrame):
    from wordcloud import WordCloud, STOPWORDS
    from PIL import Image
    #WordCloud Visualization
    text = " ".join(list(dataFrame['track_name']))
    STOPWORDS = STOPWORDS.union(["feat","Remix","Edit","Radio","Version","Mix","Remastered"])
    spotify_mask = np.array(Image.open(path.join( "spotify-logo.jpg")))
    wordcloud = WordCloud(width=2880, height=1800,background_color="white",
                          stopwords=STOPWORDS,mask = spotify_mask).generate(text)
    # Open a plot of the generated image.
    plt.figure( figsize=(10,6))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("project3_wordcloud.png")
    plt.show()

main()




