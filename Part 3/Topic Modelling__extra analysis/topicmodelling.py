# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:25:06 2017

@author: minxiaocn
"""
##anly501 project3 topic modelling
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pandas as pd

def topic_modelling():
    #data preparation
    rawdata=pd.read_csv("mySpotify.csv")
    parentcat=list(set(rawdata["parentCat"].values))
    print(parentcat)
    
    # create English stop words list
    en_stop = get_stop_words('en')
    documentlist=[]
    for pt in parentcat:
        groupdata=rawdata.loc[rawdata["parentCat"]==pt,"track_name"].tolist()
    #    groupdata_1=" ".join(groupdata)
    #    raw=groupdata_1.lower()
    #    tokens = tokenizer.tokenize(raw)
    #    stopped_tokens = [i for i in tokens if not i in en_stop]
        documentlist.append(" ".join(groupdata))
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    # create English stop words list
    en_stop = get_stop_words('en')
    
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
        
    # create sample documents
     
    # compile sample documents into a list
    doc_set = documentlist
    
    
    # list for tokenized documents in loop
    texts = []
    
    # loop through document list
    for i in doc_set:
        
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
    
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        
        # add tokens to list
        texts.append(stemmed_tokens)
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
        
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)  
    topics=ldamodel.print_topics(num_topics=5, num_words=20)
    #print the topic in documents
    print(topics)
    #print topic distribution in each documents
    for cp in corpus:
        doc_topics=ldamodel[cp]
        print(doc_topics)
        
if __name__ == "__main__":
    topic_modelling()