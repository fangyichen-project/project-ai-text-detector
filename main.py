#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mastercodes for project text detector.
Steps & Files:
Module Text Creawling (Data Input):
    -Crawler crawlt menschlich geschriebene Texte aus der Webseite von WDR und Radio Bremen und
    Texte werden durch Pipeline in Pymongo DB gespeichert.
Module Cleaning (Data Processing):
    -Data-Cleaning von Texten aus der Website von WDR.
    -Data-Cleaning von Texten aus der Website von Radio Bremen.
    -Preprocessing von menschlich geschriebenen Texten.
    -Preprocessing von ML generierten Texten. (Data vorhanden und clean.)
Module tfidf: 
    -Generieren von Tfidf-Werten von menschlich geschriebenen Texten.
    -Generieren von Tfidf-Werten von ML generierten Texten.
#TODO Module Modelling Random Forest:
    -Implementieren von Random Forest, um menschlich geschrieben und ML generierten Texten
    zu klassifizieren. Ergebnisse werden ausgewertet.
Module Modelling SVM:
    -Implementieren von SVM, um menschlich geschrieben und ML generierten Texten
    zu klassifizieren. Ergebnisse werden ausgewertet.
#TODO Module Modelling Word Embedding:
    -Implementieren von Word Embedding, um menschlich geschrieben und ML generierten Texten
    zu klassifizieren. Ergebnisse werden ausgewertet. 
#TODO Module Resules
   -Vergleich von den Ergebnisse aller drei Modellen.
    
@author: fangyichen


"""

#%%
import sys
import os
import subprocess
import pandas as pd
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


#Module Text Crawling. 
#Texte aus der Webseite crawlen.
#Output: Human written texts from wdr and radio bremen stored in mongo db.
sys.path.append('//Users/fangyichen/Project Text Detector的副本/dlf_spider/bremenspider/spiders')
import brmspider
import wdrspider

#Call a function to crawl texts from the website of wdr.
wdrspider.run_wdr_spider()

#Call a function to crawl texts from the website of radio bremen.
brmspider.run_brm_spider()

#%%
###Module Cleaning and Preprocessing of human written texts.

#Input: Crawled texts stored in mongo db.

#Data-Cleaning von Texten aus der Website von WDR.
#Output: Cleaned human written texts from WDR.
# Import the file (datacleansing_human_text_wdr.py)
import datacleansing_human_text_wdr
# Call a function from the imported module
cleaned_human_text_wdr = datacleansing_human_text_wdr.cleaned_human_text_wdr()

#%%
#Data-Cleaning von Texten aus der Website von Radio Bremen. 
#Input: Texte from Radio
#Output: Cleaned human written texts from Radio Bremen.
# Import the file (datacleansing_human_text_bremen.py)
import datacleansing_human_text_bremen
# Call a function from the imported module
cleaned_human_text_bremen = datacleansing_human_text_bremen.cleaned_human_text_bremen()

#%%
#Put the cleaned human written texts together. Preprocess the human written texts.
#Inputs: Preprocessed human written texts from WDR and Radio Bremen together stored in pymongo.
#Outputs include 5 columns: 
    #text_without_stopword
    #stemmed_text_with_stopword      
    #stemmed_text_without_stopword    
    #lemmatized_text_with_stopword    
    #lemmatized_text_without_stopword 
#Name of collection: 'human_written_texts(preprocessed)'
#Database_name = "scrapy_data"

#Import the file (preprocess_human_written_text.py)
import preprocess_human_written_text

'''
#Call a function from the imported module to upload the preprocessed texts into pymongo.
preprocess_human_written_text.create_new_collection_human()
'''
#Call a function from the imported module to update the preprocessed human written texts in pymongo.
preprocess_human_written_text.update_collection_human()

#%%
###Module Cleaning and Preprocessing of ML generated texts.

#Input: existed and cleaned ml texts stored in Pymongo db.
#Output: preprocessed ml texts stored in pymongo db.
#Name of collection: 'ml_texts(preprocessed)'
#Database_name = "scrapy_data"
#Outputs include 6 columns: 
    #cleaned_text
    #cleaned_text_without_stopword
    #stemmed_text_with_stopword      
    #stemmed_text_without_stopword    
    #lemmatized_text_with_stopword    
    #lemmatized_text_without_stopword 
    
#Import the file (preprocess_ml_text.py)
import preprocess_ml_text

'''
#Call a function from the imported module to upload the preprocessed ml-texts in pymongo.
preprocess_ml_text.create_new_collection_ml()

#Call a function from the imported module to update the preprocessed ml generated texts in pymongo.
preprocess_ml_text.update_collection_ml()
'''
#%%
###Module tfidf
#split data in traning and testing datasets.
#label and generate tfidf-values of human written and ml generated texts (stemmatized texts without stopwords).
#Input: human written and ml generated texts (stemmatized texts without stopwords) in mongo.
#Output:
    #x_train_tfidf: tfidf values of training input data in matrix.
    #x_test_tfidf: tfidf values of testing input data in matrix.
    #y_train: training prediction data
    #y_test: testing prediction data
    #x_train_df: tfidf values of training input data and features in one dataframe.
    #features
    #x: Input data
    #y: Prediction data
    
#Import the file (text_tfidf_vectorizing.py)
import text_tfidf_vectorizing

#Call a function from the imported module
text_tfidf_vectorizing.text_tfidf_vectorizing()

#%%
###Module support vector machine
#Classification using svm.
#Input: Output from module tdidf
#Output:
    #Balanced Accuracy
    #Confusion Matrix from a K-Fold Cross-Validation
    #Accuracy Score
    #Results of k-fold cross-validation
#Import the file (support_vector_machine.py)
import text_tfidf_vectorizing

#Call a function from the imported module
text_tfidf_vectorizing.svm()

#%%
#TODO Spagthetti Codes löschen u. anschließen.
###Module random forest
#Classification using random forest.










