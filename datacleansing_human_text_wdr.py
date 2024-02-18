#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 01:40:32 2023

@author: fangyichen

Import mongo data(crawled from wdr) in pandas, clean the data and update it in mongo.

"""
#%%
# Import mongo data into panda dataframe (df).

from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import re


uri = "mongodb://localhost:27017"
database_name = "scrapy_data"
collection_name = "human_written_texts(wdr)"

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

df.info()

print(df.head())


#%%
# Delete "\n"

# Define the strings to filter
strings_to_filter = ["\n", 
                     "Unsere Quelle:  Über dieses Thema berichtet der WDR am 28.12.2023 auch im  WDR -Fernsehen in der Lokalzeit aus Dortmund und im Radio auf  WDR2 . ",
                     "Quelle:",
                     "Quelle: dpa",
                     "Quelle: red/dpa",
                     "Quelle: dpa/red",
                     "Quelle: sid",
                     "Quelle: sid ",
                     "Quelle: wdr",
                     "Quelle: WDR"
                     " Quelle:   -Reporter vor Ort"
                     "Quelle:   -Reporter vor Ort"]


def remove_strings(text):
    for string in strings_to_filter:
        text = text.replace(string, "")
    return text

# Applying the function to the specified column
df['text'] = df['text'].apply(remove_strings)

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

#%%
#Delete text that is not german in the data set.
from langdetect import detect

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'  # Handle cases where language detection fails

# Apply language detection to the 'text' column and create a new column 'language'
df['language'] = df['text'].apply(detect_language)

# Filter out rows where the detected language is not German ('de')
df = df[df['language'] == 'de']

# Drop the language column, because all the texts are in German.
df = df.drop('language', axis=1)

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
#Define the output (cleaned human written texts from wdr).
def cleaned_human_text_wdr():
    return df

if __name__ == "__main__":
    cleaned_human_text_wdr()


