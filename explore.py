#################################################################################################################################
#import libraries
import pandas as pd
import numpy as np
import re
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import nltk.sentiment
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from requests import get
from bs4 import BeautifulSoup
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import re
import time

#################################################################################################################################
#Train-Test Split

def split_minecraft_data(df):
    '''
    This function performs split on minecraft repo data, stratified on language.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.language)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.language)
    return train, validate, test

#call it with: train, validate, test = split_minecraft_data(df)

################################################################################################################################

#Visualizations

def get_language_freq(train):
    '''
    This function takes in the training data set and creates a countplot
    utilizing Seaborn to visualize the range and values of programming
    languages in GitHub Repositories'''
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    cpt = sns.countplot(x='language', data=train, palette='GnBu')
    plt.title('Java is the Most Common Language in our Dataset')
    plt.xlabel("Programming Language")
    plt.ylabel('Count of Languages')
    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
    plt.show()
    
def get_wordcount_bar(train):
    '''
    This function takes in the training dataset and creates a bar plot of the
    average wordcount of repository based on their language type
    '''
    #Make a column on the df for word count
    train['word_count'] = train.lemmatized.str.split().apply(len)
    #Use groupby to get an average length per language
    language_wordcount = train.groupby('language').word_count.mean().sort_values(ascending=False)
    #Set style, make a chart
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    ax = sns.barplot(x=language_wordcount.values, 
                 y=language_wordcount.index, palette='Set3')
    plt.title('Average Wordcount of Languages in Readme Files')
    plt.xlabel("Average Word Count")
    plt.ylabel('Language')
    plt.show()

def get_top10_python(train):
    '''
    This function takes in a train dataset, creates a df for
    the Python coding language, then plots the 10 most frequently
    used words in the column labeled 'language'
    '''
    python_df = train[train['language']=='Python']
    python_txt = ' '.join(python_df['lemmatized'])
    python_txt = pd.Series(python_txt.split()).value_counts().head(10)
    
    sns.set_style("darkgrid")
    python_txt.sort_values(ascending=False).plot.barh(color='#4B8BBE', width=.9, figsize=(10, 6))

    plt.title('10 Most frequently occuring Python strings')
    plt.ylabel('Strings')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = python_txt.reset_index()['index']
    _ = plt.yticks(ticks, labels)
    
    plt.show()


def get_top10_java(train):
    '''
    This function takes in a train dataset, creates a df for
    the Java coding language, then plots the 10 most frequently
    used words in the column labeled 'language'
    '''
    java_df = train[train.language == 'Java']
    java_txt = ' '.join(java_df['lemmatized'])
    java_txt = pd.Series(java_txt.split()).value_counts().head(10)
    
    sns.set_style("darkgrid")
    java_txt.sort_values(ascending=False).plot.barh(color='#f89820', width=.9, figsize=(10, 6))

    plt.title('10 Most frequently occuring Java strings')
    plt.ylabel('Strings')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = java_txt.reset_index()['index']
    _ = plt.yticks(ticks, labels)
    
    plt.show()


def get_top10_js(train):
    '''
    This function takes in a train dataset, creates a df for
    the JavaScript coding language, then plots the 10 most frequently
    used words in the column labeled 'language'
    '''
    js_df = train[train.language == 'JavaScript']
    js_txt = ' '.join(js_df['lemmatized'])
    js_txt = pd.Series(js_txt.split()).value_counts().head(10)
    
    sns.set_style("darkgrid")
    js_txt.sort_values(ascending=False).plot.barh(color='#F0DB4F', width=.9, figsize=(10, 6))

    plt.title('10 Most frequently occuring JavaScript strings')
    plt.ylabel('Strings')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = js_txt.reset_index()['index']
    
    plt.show()