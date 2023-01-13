#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd


# In[2]:


def basic_clean(string):
    '''
    This function accepts a string as an input
    then lowercases everything, normalizes unicode
    characters, and replaces anything that is
    not a letter, number, whitespace, 
    or a single quote.
    '''
    cleaned = string.lower()
    cleaned = unicodedata.normalize('NFKD', cleaned)    .encode('ascii', 'ignore')    .decode('utf-8', 'ignore')
    cleaned = re.sub(r"[^a-z0-9'\s]", '', cleaned)
    
    return cleaned


# In[3]:


def tokenize(string):
    '''
    This function takes in a string as an input
    then tokenizes all words in the string.
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string, return_str=True)


# In[4]:


def stem(string):
    '''
    This function takes in a string as an input
    then stems all words in the string.
    '''
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string_stemmed = ' '.join(stems)
    return string_stemmed


# In[5]:


def lemmatize(string):
    '''
    This function takes in a string as an input
    then lemmatizes all words in the string.
    '''
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string_lemmatized = ' '.join(lemmas)

    return string_lemmatized


# In[6]:


def remove_stopwords(string, extra_words=[], exclude_words=[]):
    '''
    This function takes in a string as an input
    then removes stopwords. The function has two
    additional parameters that define additional
    stopwords to remove in extra_words as a list,
    and defines stopwords to exclude from removal
    in exlude_words as a list. extra_words and
    exclude_words are empty lists by default.
    '''
    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    stopword_list = stopword_list.union(set(extra_words))
    
    words = string.split()
    
    filtered_words = [w for w in words if w not in stopword_list]

    #print('Removed {} stopwords'.format(len(words) - len(filtered_words)))
    #print('---')

    string_without_stopwords = ' '.join(filtered_words)

    return string_without_stopwords


# In[7]:
extra_stops = ['server', 'run', '&#9;', "' ", " '", "'",'Minecraft','minecraft','minecraft ',' minecraft', 'abstract','and','arguments','assert','break','byte','case','char','class',
               'const','continue','default','double','else','enum','extends','false','final','finally','float','for',
               'goto','if','implements','import','in','instanceof','int','interface','long','native','new','null',
               'package','pass','private','protected','public','raise','return','short','static','super','switch',
               'synchronized','this','throw','throws','transient','true','try','void','volatile','while','with',
               'yield', 'http', 'com', 'github', 'www', 'version', 'file']

def prep_readme_data(df, column, extra_words=extra_stops, exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the repo name, original text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df = df.dropna()
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                  extra_words=extra_words,
                                  exclude_words=exclude_words)
    
    
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    df = map_other_languages(df)
    
    return df


#
def map_other_languages(df):
    '''
    This function takes in a df with 'languages' column
    containing the coding language of the repo. Any language
    that is not Python, Java, or JavaScript will be marked
    as 'Other'
    '''
    top_languages = ['Python', 'Java', 'JavaScript']
    df.loc[~df['language'].isin(top_languages), 'language'] = 'Other'
    
    return df