#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:10:43 2023

@author: fangyichen


"""
# %%
#Import ML_df
import re
import pandas as pd
import nltk 
import string
from nltk.stem.snowball import SnowballStemmer
from pymongo import MongoClient

# %%
#Import ML_df in german.
file_path = '/Users/fangyichen/Project Text Detector的副本/multitude.csv'

df = pd.read_csv(file_path)

#filter the data set to get german and ml text
ml_df = df[(df['language'] == 'de') & (df['label'] == 1)]

'''
ml_df.info()
'''

# %%
#Text cleaning

def text_clean(text_list):
    cleaned_texts = []
    for text in text_list:
        # Lowercasing
        text = text.lower()
        # Removing punctuation
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        text = re.sub(r'\d+', 'num', text)
        # Replace numbers with "num".
        text = re.sub(r'\d+', 'num', text)
        cleaned_texts.append(text)  # Append the cleaned text to a new list
    return cleaned_texts  # Move this outside the loop to return after all texts are processed

ml_df["cleaned_text"] = text_clean(ml_df["text"])

# %%
#Remove specific characters

def remove_specific_characters(text_list):
    cleaned_texts = []
    for text in text_list:
        # Define the characters you want to remove
        characters_to_remove = ['–', '“', '„', '”']   
        # Create a regular expression pattern to match these characters
        pattern = '|'.join(map(re.escape, characters_to_remove))
        # Remove the characters using the pattern
        text = re.sub(pattern, '', text)
        cleaned_texts.append(text)
    return cleaned_texts

ml_df["cleaned_text"] = remove_specific_characters(ml_df["cleaned_text"])

# %%
#Remove stopwords.

#Stop words present in the library.
#Add extra stop words in the list.
stopwords = nltk.corpus.stopwords.words('german')

# Read the contents of the text file
with open('german_stopwords_full.txt', 'r') as file:
    text = file.read()

# Convert the text into a list of words
stopword_list = text.split()
stopword_full = stopword_list + stopwords 


def remove_stopwords(words_list, stopword_full):
    return [word for word in words_list if word not in stopword_full]

# Apply the function to the 'cleaned_text' column
ml_df['cleaned_text_without_stopword'] = ml_df['cleaned_text'].apply(lambda x: ' '.join(remove_stopwords(x.split(), stopword_full)))

# %%
#Tokenize ml_df['cleaned_text_without_stopwords'] and ml_df['cleaned_text'].

#defining function for tokenization
def tokenization(text):
    tokens = re.split('\W+', text)
    return tokens

ml_df['cleaned_text_without_stopword'] = ml_df['cleaned_text_without_stopword'].apply(lambda x: tokenization(x))
ml_df['cleaned_text'] = ml_df['cleaned_text'].apply(lambda x: tokenization(x))

#print(ml_df.head())


# %%
##Till here, all the processes are applied to text. Now we seperate and add 4 rows:
#1.Stemmed text with stopwords
#2.Stemmed of text without stopwords
#3.Lemmatized text with stopwords
#4.Lemmatized text without stopwords

# %%
#Stemming of text with stopwords

stemmer = SnowballStemmer("german")

#defining a function for stemming
def stemming(text):
    stem_text = [stemmer.stem(word) for word in text]
    return stem_text

#applying the function
ml_df['stemmed_text_with_stopword']= ml_df['cleaned_text'].apply(lambda x: stemming(x))


# %%
#Stemming of text without stopwords
ml_df['stemmed_text_without_stopword']= ml_df['cleaned_text_without_stopword'].apply(lambda x: stemming(x))

# %%                     
#Lemmatizing of text with stopwords.                                                                    
from nltk.stem import WordNetLemmatizer

#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

#applying the function
ml_df['lemmatized_text_with_stopword']= ml_df['cleaned_text'].apply(lambda x: stemming(x))

# %%
#lemmatizing of text without stopwords
ml_df['lemmatized_text_without_stopword']= ml_df['cleaned_text_without_stopword'].apply(lambda x: stemming(x))

# %%
ml_df.info()
ml_df.drop(columns=['length','label','multi_label','split','language','length'], inplace=True)

# %%

#Define the function (upload the data into mongo db and create a new collection for it).
def create_new_collection_ml():
    #Upload the data into mongo db and create a new collection for it.
    
    uri = "mongodb://localhost:27017"
    database_name = "scrapy_data"
    
    mongo_client = MongoClient(uri)
    database = mongo_client[database_name]
    
    new_collection = database['ml_texts(preprocessed)']
    new_collection.insert_many(ml_df.to_dict('records'))

if __name__ == "__main__":
    create_new_collection_ml()

# %%
#Define the function of update the pymongo collection with preprocessed texts.

def update_collection_ml():
    #Update the data collection after change.
    
    uri = "mongodb://localhost:27017"
    database_name = "scrapy_data"
    
    mongo_client = MongoClient(uri)
    database = mongo_client[database_name]
    
    new_collection = database['ml_texts(preprocessed)']
    
    for index, row in ml_df.iterrows():
        filter_query = {"text": row["text"]}
        update_query = {"$set": {"text": row["text"],
                                 "cleaned_text": row["cleaned_text"],
                                 "cleaned_text_without_stopword": row["cleaned_text_without_stopword"],
                                 "stemmed_text_with_stopword": row["stemmed_text_with_stopword"],
                                 "stemmed_text_without_stopword": row["stemmed_text_without_stopword"],
                                 "lemmatized_text_with_stopword": row ["lemmatized_text_with_stopword"],
                                 "lemmatized_text_without_stopword": row ["lemmatized_text_without_stopword"]
                                 }}
        
        new_collection.update_many(filter_query, update_query)
    
if __name__ == "__main__":
    update_collection_ml()


