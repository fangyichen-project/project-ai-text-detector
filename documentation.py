#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sun Dec 24 01:34:27 2023
Documentation

@author: fangyichen

#mongodb:
#mongo data set: 'scrapy_data'
#backupcollection for texts from radio bremen: 'scrapy_data.backup_human_written_texts'
#processes collection for texts from radio bremen: 'scrapy_data.human_written_texts'
#backupcollection for texts from wdr: 'scrapy_data.backup_human_written_texts(wdr)'
#processes collection for texts from wdr: 'scrapy_data.human_written_texts(wdr)'

#crawler 1
#BOT_NAME = "dlf_spider"
#SPIDER_MODULES = ["bremenspider.spiders"]
#NEWSPIDER_MODULE = "bremenspider.spiders"
#Spider Name: "bremen_spider_text"
#Pipeline to Mongo: MongoDBPipeline

#crawler 2
#BOT_NAME = "dlf_spider"
#SPIDER_MODULES = ["bremenspider.spiders"]
#NEWSPIDER_MODULE = "bremenspider.spiders"
#Spider Name: "second_spider"
#Pipeline to Mongo: MongoDBPipeline2

#venv:
    nltk-env for general cases 
    venv2 for word embedding


#data set:
    #ml data from a Open Source project multiverse
    #human written text data from radio bremen (crawler 1) and wdr (Crawler 2)
    
#The text get preprocessed and vektorized. I want to test four possibilitties:
    stemmed_text_with_stopword': 1,
    'stemmed_text_without_stopword': 1,
    'lemmatized_text_with_stopword': 1,
    'lemmatized_text_without_stopword': 1
   
#x: tfidf 
#y: label 0 for human and 1 for ml.

TODO
# random forest. delete spaghetti codes.



'''