#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1.label ml and human texts.
2.put the datasets together.
3.get tfidf values.
4.split data sets into training and testing data sets.

@author: fangyichen
"""

# %%
import re
from pymongo import MongoClient
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy import sparse
import string as str
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %%
#Import stemmatized texts without stopwords.

#Function to access data from the mongo db and write them into df.
def unwind_text_field(collection_name,df):
    uri = "mongodb://localhost:27017"
    database_name = "scrapy_data"
    mongo_client = MongoClient(uri)
    database = mongo_client[database_name]
    collection = database[collection_name]
    # Unwind the text field
    results = collection.aggregate([
      {
        "$project": {
      'stemmed_text_without_stopword': 1 }
        }]
    )
    
    for result in results:
        df.append(result)
    
    return df

#Apply the function for human written texts.
human_written_texts = []
hm_collection_name = "human_written_texts(preprocessed)"
human_written_texts = unwind_text_field(hm_collection_name,human_written_texts)
human_written_texts = pd.DataFrame(human_written_texts)

#Apply the function for ml texts.
ml_written_texts = []
ml_collection_name = "ml_texts(preprocessed)"
ml_written_texts = unwind_text_field(ml_collection_name,ml_written_texts)
ml_written_texts = pd.DataFrame(ml_written_texts)

# %%
#Label human_text as 0 a. ml_text as 1.
human_written_texts["label"] = 0
ml_written_texts["label"] = 1

#Combine two dfs and reset index.
combined_df = pd.concat([human_written_texts, ml_written_texts], axis=0, ignore_index=True)

# %%
#Remove "num". Remove intergers.
def remove_int(column):
    column1 = []
    for text_list in column:
        text_list = [re.sub('num', '', item) for item in text_list]
        text_list = [re.sub(r'\b\d+\b', '', item) for item in text_list]
        column1.append(text_list)
    return column1  # Return the modified column

# Assuming combined_df is a DataFrame with a column named "stemmed_text_without_stopword"
# Replace digits in the specified column
combined_df["stemmed_text_without_stopword"] = remove_int(combined_df["stemmed_text_without_stopword"])

# %%
#Turn the list of words in the "text" column into text for tfidf.
def preprocess_text_column(df, column_name):
    # Join text values in the column into a list of strings
    df[column_name] = df[column_name].apply(lambda x: " ".join(x))
    # Turn the column of the dataframe to a list
    text_list = df[column_name].tolist()
    return text_list

combined_df["text"]=preprocess_text_column(combined_df, "stemmed_text_without_stopword")

# %%
#Split the data into training and testing data sets.
x = combined_df["text"]
y = combined_df["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 4)

# %%
#TF-IDF Vectorization of x_train and x_test. 
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

#Put training and testing dataset together.
x = vstack((x_train_tfidf, x_test_tfidf)) 
# %%
#Get tf-idf-features.
features = vectorizer.get_feature_names_out()
x_train_df = pd.DataFrame(x_train_tfidf.toarray(), columns=features)
x_test_df = pd.DataFrame(x_test_tfidf.toarray(), columns=features)

#%%
#Define the output.
def text_tfidf_vectorizing():
    return x_train_tfidf, x_test_tfidf, y_train, y_test, x_train_df, features, x, y

if __name__ == "__main__":
    text_tfidf_vectorizing()
    
    

