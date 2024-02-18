#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#Perprocess and vektorize the text
#lowercasing, removing punctuation and stopwords and tokenized. 
#Stemming/lemmatization, with and without stopwords.
"""

from pymongo import MongoClient
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer

#%%
# Import cleaned human written texts from radio bremen and wdr. Put them together.

# Import the module (datacleansing_human_text_wdr.py)
import datacleansing_human_text_wdr
# Call a function from the imported module
cleaned_human_text_wdr = datacleansing_human_text_wdr.cleaned_human_text_wdr()

# Import the module (datacleansing_human_text_bremen.py)
import datacleansing_human_text_bremen
# Call a function from the imported module
cleaned_human_text_bremen = datacleansing_human_text_bremen.cleaned_human_text_bremen()

#Put both df together.
preprocess_df = pd.concat([cleaned_human_text_wdr, cleaned_human_text_bremen])

preprocess_df.info()

#change the display in console to show whole content in a row
pd.set_option('display.max_columns', None)

# %%
#LowercasingPunctuation Removal 

#Import library for punctuation
import string
string.punctuation

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#storing the puntuation free text
preprocess_df['text']= preprocess_df['text'].apply(lambda x:remove_punctuation(x))

# %%
#Lowering the Text 
preprocess_df['text']= preprocess_df['text'].apply(lambda x:x.lower())

# %%
#Tokenization. Sentences are tokenized into words.

#defining function for tokenization
def tokenization(text):
    tokens = re.split('\W+', text)
    return tokens

preprocess_df['text'] = preprocess_df['text'].apply(lambda x: tokenization(x))

# %%
#Stop Word Removal.

#Stop words present in the library.
#Add extra stop words in the list.
stopwords = nltk.corpus.stopwords.words('german')

# Read the contents of the text file
with open('german_stopwords_full.txt', 'r') as file:
    text = file.read()

# Convert the text into a list of words
stopword_list = text.split()

stopword_full = stopword_list + stopwords 

def remove_stopwords_from_column(df, column_name, column_name_2):
    df[column_name_2] = df[column_name].apply(lambda text: ' '.join([word for word in text if word not in stopword_full]) if text else '')
    return df

remove_stopwords_from_column(preprocess_df,'text','text_without_stopword')

preprocess_df['text_without_stopword'] = preprocess_df['text_without_stopword'].apply(lambda x: tokenization(x))

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
preprocess_df['stemmed_text_with_stopword']= preprocess_df['text'].apply(lambda x: stemming(x))

# %%
#Stemming of text without stopwords
preprocess_df['stemmed_text_without_stopword']= preprocess_df['text_without_stopword'].apply(lambda x: stemming(x))

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
preprocess_df['lemmatized_text_with_stopword']= preprocess_df['text'].apply(lambda x: stemming(x))

# %%
#lemmatizing of text without stopwords
preprocess_df['lemmatized_text_without_stopword']= preprocess_df['text_without_stopword'].apply(lambda x: stemming(x))

# %%
preprocess_df.info()

# %%
#Define the function of upload the preprocessed texts into pymongo and create a new collection.

def create_new_collection_human():
    #Upload the data into mongo db and create a new collection for it.
    uri = "mongodb://localhost:27017"
    database_name = "scrapy_data"

    mongo_client = MongoClient(uri)
    database = mongo_client[database_name]

    new_collection = database['human_written_texts(preprocessed)']
    new_collection.insert_many(preprocess_df.to_dict('records'))

if __name__ == "__main__":
    create_new_collection_human()
    
    
# %%
#Define the function of update the pymongo collection with preprocessed texts.

def update_collection_human():
    #Update the data collection after change.
    
    uri = "mongodb://localhost:27017"
    database_name = "scrapy_data"
    
    mongo_client = MongoClient(uri)
    database = mongo_client[database_name]
    
    new_collection = database['human_written_texts(preprocessed)']
    
    for index, row in preprocess_df.iterrows():
        filter_query = {"_id": row["_id"]}
        update_query = {"$set": {"text": row["text"],
                                 "text_without_stopword": row["text_without_stopword"],
                                 "stemmed_text_with_stopword": row["stemmed_text_with_stopword"],
                                 "stemmed_text_without_stopword": row["stemmed_text_without_stopword"],
                                 "lemmatized_text_with_stopword": row ["lemmatized_text_with_stopword"],
                                 "lemmatized_text_without_stopword": row ["lemmatized_text_without_stopword"]
                                 }}
        
        new_collection.update_many(filter_query, update_query)
    
if __name__ == "__main__":
    update_collection_human()
