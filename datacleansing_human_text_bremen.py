#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:03:38 2023

@author: fangyichen
Import mongo data in pandas, clean the data and update it in mongo.

"""
#%%
# Import mongo data into panda dataframe (df).

from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import re


uri = "mongodb://localhost:27017"
database_name = "scrapy_data"
collection_name = "human_written_texts"

mongo_client = MongoClient(uri)
database = mongo_client[database_name]
collection = database[collection_name]

# Unwind the text field
results = collection.aggregate([
  {
    "$unwind": "$text"
  }
])

df = pd.DataFrame(list(results))


#%%
# Delete "Radio Bremen", "\nDiepenau 10", "\n28195 Bremen", "\n"

# Define the strings to filter
strings_to_filter = ["Radio Bremen", "\nDiepenau 10", "\n28195 Bremen", "\n"]

# Filter the DataFrame using isin() to create a boolean mask
mask = df['text'].isin(strings_to_filter)

# Invert the mask to select rows NOT matching the conditions and keep them in the DataFrame
df = df[~mask]

#%%
#Delete duplicate texts.

df.drop_duplicates(subset='text', keep='first', inplace=True)

#%%

# From the text with the same _id, keep 12 sentences ending with point. 

# Group the texts with the same id in the order in df.
df = df.groupby('_id')['text'].agg(lambda x: ' '.join(x)).reset_index()

#change the display in console
pd.set_option("max_colwidth", 3000)

# Filter out text with less than 3 periods, because too short.
df['text'] = df['text'].apply(lambda x: '.'.join(x.split('.')[:12]) if x.count('.') >= 11 else x)

#test and see the processed data
print(df[["text"]].iloc[:5])

#%%
# Find out the information of the data frame
df.info()

# Calculate word counts of text and plot a histogram
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

# Plotting the histogram
plt.hist(df['word_count'], bins=10, alpha=0.7, color='blue')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Human Written Word Counts')
plt.show()

#%%
#Define the output (cleaned human written texts from radio bremen).
def cleaned_human_text_bremen():
    return df

if __name__ == "__main__":
    cleaned_human_text_bremen()



