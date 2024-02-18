#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 01:37:01 2024

Word Embedding 
@author: fangyichen
"""
import numpy as np
import pandas as pd
import nltk
import sys
from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pymongo import MongoClient

from sklearn.model_selection import train_test_split
import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

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
      'text': 1 }
        }]
    )
    
    for result in results:
        df.append(result)
    
    return df

#Apply the function for human written texts from brm.
human_written_texts_brm = []
hm_collection_name = "human_written_texts(cleaned)"
human_written_texts_brm = unwind_text_field(hm_collection_name,human_written_texts_brm)
human_written_texts_brm  = pd.DataFrame(human_written_texts_brm)

#Apply the function for human written texts from wdr.
human_written_texts_wdr = []
hm_collection_name = "human_written_texts(wdr)"
human_written_texts_wdr = unwind_text_field(hm_collection_name,human_written_texts_wdr)
human_written_texts_wdr = pd.DataFrame(human_written_texts_wdr)

#Combine df for human written texts.
human_written_texts = pd.concat([human_written_texts_wdr, human_written_texts_brm], ignore_index=True)

#Apply the function for ml texts.
ml_written_texts = []
ml_collection_name = "ml_texts(preprocessed)"
ml_written_texts = unwind_text_field(ml_collection_name,ml_written_texts)
ml_written_texts = pd.DataFrame(ml_written_texts)

# %%
#Label human_text as 0 a. ml_text as 1.
human_written_texts["label"] = 0
ml_written_texts["label"] = 1

all_texts = pd.concat([human_written_texts, ml_written_texts], ignore_index=True)
all_texts = all_texts.drop("_id", axis=1)

#Change text into a list and change the elements to string.
text = all_texts['text'].tolist()
text = [str(element) for element in text]

# %%

all_texts['text'] = all_texts['text'].astype(str)

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(all_texts, test_size=0.2, random_state=12)

# Create TensorFlow datasets
batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((train_df['text'].values, train_df['label'].values))
#train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_tensor_slices((val_df['text'].values, val_df['label'].values))
#val_ds = val_ds.batch(batch_size)


# %%
# Create a TextVectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=1000,  # Maximum vocabulary size
    output_mode='int',  # Output integer token indices
)

# Make a text-only dataset (without labels), then call adapt on the list of 
#strings to create the vocabulary.
train_text = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Map strings to integers.
def vectorize_text(text):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text)

# %%
# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(all_texts, test_size=0.2, random_state=12)

# Create TensorFlow datasets
batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((train_df['text'].values, train_df['label'].values))
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_tensor_slices((val_df['text'].values, val_df['label'].values))
val_ds = val_ds.batch(batch_size)

# Create a TextVectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=1000,  # Maximum vocabulary size
    output_mode='int',  # Output integer token indices
)

# Adapt the TextVectorization layer on the training text
train_text = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Map strings to integers.
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Apply the TextVectorization layer to the datasets
train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)

# %%
embedding_dim = 16


