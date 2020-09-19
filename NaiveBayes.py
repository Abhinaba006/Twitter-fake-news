# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 06:59:27 2020

@author: Abhinaba Das
"""
import numpy as np
import pandas as pd 
from collections import Counter 
import re, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
lem = WordNetLemmatizer()
stemmer = PorterStemmer()
lis = []

class Naive_Bayes():
     '''
         this will make your text model ready. feed it coloumn only using apply and
         will return the columns as needed
     '''
     def __init__(self, gram, prob):
        self.true_counter = []
        self.fake_counter = []
        self.true_word_len = 0
        self.fake_word_len = 0
        self.gram = gram
        self.prob = prob
        
     def clean(self, text):
        
            
        # Removing special syntax
       text = re.sub(r"(b')+" , "" , text)
    
        # Removing URls
       text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(https?://[^\s]+))' , "" , text)
       text = re.sub(r'http\S+' , "" , text)
    
        # Removing Usernames
       text = re.sub(r'@[^\s]+' , "" , text)
    
        # Removing Hashtags
       text = re.sub(r'#([^\s]+)' , r'\1' , text)
    
        # Removing HTML Tags
       text = re.sub(r'<.*?>' , "" , text)
    
        # Removing Emogis
       emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
       text = emoji_pattern.sub(r'', text) 
    
        # Removing special Emogis
       text = (text.
                replace('\\xe2\\x80\\x99', "'").
                replace('\\xc3\\xa9', 'e').
                replace('\\xe2\\x80\\x90', '-').
                replace('\\xe2\\x80\\x91', '-').
                replace('\\xe2\\x80\\x92', '-').
                replace('\\xe2\\x80\\x93', '-').
                replace('\\xe2\\x80\\x94', '-').
                replace('\\xe2\\x80\\x94', '-').
                replace('\\xe2\\x80\\x98', "'").
                replace('\\xe2\\x80\\x9b', "'").
                replace('\\xe2\\x80\\x9c', '"').
                replace('\\xe2\\x80\\x9c', '"').
                replace('\\xe2\\x80\\x9d', '"').
                replace('\\xe2\\x80\\x9e', '"').
                replace('\\xe2\\x80\\x9f', '"').
                replace('\\xe2\\x80\\xa6', '...').
                replace('\\xe2\\x80\\xb2', "'").
                replace('\\xe2\\x80\\xb3', "'").
                replace('\\xe2\\x80\\xb4', "'").
                replace('\\xe2\\x80\\xb5', "'").
                replace('\\xe2\\x80\\xb6', "'").
                replace('\\xe2\\x80\\xb7', "'").
                replace('\\xe2\\x81\\xba', "+").
                replace('\\xe2\\x81\\xbb', "-").
                replace('\\xe2\\x81\\xbc', "=").
                replace('\\xe2\\x81\\xbd', "(").
                replace('\\xe2\\x81\\xbe', ")")
                     )    
    
        # Lower and stopwords removal
       text = " ".join([word for word in text.lower().split() if not word in set(stopwords.words('english'))])
    
        # Punctuation Removal
       text = "".join([char if char not in string.punctuation else ' ' for char in text])
    
        # Number Removal
       text = re.sub(r"[^A-Z a-z]" , "" , text)
    
        # Emogi Residual Removal
       text = re.sub(r"x[a-z]+ " , "" , text)
    
        # Removing words or length less than 2
       text = " ".join([word for word in text.split() if len(word)>2])
    
        # Removing double or trailing spaces
       text = " ".join(text.split())
        
        # Similar word removal
       corona_similar = ['novelcoronavirus' , 'covid19' , 'covid' , 'corona' , 'coronavirus']
       india_similar = ['indian']
       for word in corona_similar:
           text = re.sub(word , "corona" , text)
       for word in india_similar:
            text = re.sub(word , "india" , text)
    
        # Lemmatize
       lemmatizer = WordNetLemmatizer()
       text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        
       return text
                           

     def pre(self, data):
         data.tweet = data.tweet.apply(self.clean)
         data.tweet = data.tweet.apply(self.Ngram)
          
     def counter(self, data):
         true_df = data[data['target(fake=0)']==1]
         fake_df = data[data['target(fake=0)']==0]
         
         
         if self.prob == 'word':
             
             true_words  = list(true_df['tweet'].sum())
             self.true_counter = Counter(true_words)
             self.true_word_len = len(true_words)
             
             fake_words  = list(fake_df['tweet'].sum())
             self.fake_counter = Counter(fake_words)
             self.fake_word_len = len(fake_words)
             
         elif self.prob == 'sentence':
             
             true_df = true_df.tweet.apply(self.seta)
             fake_df = fake_df.tweet.apply(self.seta)
             print(true_df)
             
             true_words  = list(true_df.sum())
             self.true_counter = Counter(true_words)
             self.true_word_len = len(true_df)
            
             fake_words  = list(fake_df.sum())
             self.fake_counter = Counter(fake_words)
             self.fake_word_len = len(fake_df)
          
     
    
     def Ngram(self, text):
         text = text.split()
         if self.gram == 1:
             return text
         elif self.gram == 2:
             return [' '.join([text[i], text[i+1]]) for i in range(0, len(text)-1)]
         elif self.gram == 3:
             lis = []
             for i in range(0, len(text)-1):
                 lis = []
                 lis.append(text[i])
                 lis.append(' '.join([text[i], text[i+1]]))
             return lis
             
         
     def seta(self, text):
         return list(set(text))
            
            
     def predict(self, df):
        y = []
        true_prob = self.true_word_len / (self.true_word_len + self.fake_word_len + 1)
        fake_prob = 1 - true_prob
        true_voc_count = self.true_counter
        fake_voc_count = self.fake_counter
        for i in range(len(df)):
            ul_true = np.log(true_prob)
            ul_fake = np.log(fake_prob)
            for word in df.iloc[i].tweet:
                ul_true += np.log(((true_voc_count[word]+ 1) ))
                ul_fake += np.log(((fake_voc_count[word]+ 1) ))
            if ul_true >= ul_fake:
                y.append(1)
            else:
                y.append(0)
        return y
            
            
