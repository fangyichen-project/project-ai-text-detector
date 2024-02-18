# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

#Pipeline to Mongo

from scrapy import Item
import pymongo
from itemadapter import ItemAdapter

# Pipeline for crawler for radio bremen
class MongoDBPipeline(object):
    @classmethod
    def from_crawler(cls, crawler):
        cls.DB_URL = crawler.settings.get('MONGO_DB_URI')
        cls.DB_NAME = crawler.settings.get('MONGO_DB_NAME')
        cls.collection_name = crawler.settings.get('MONGO_COLLECTION_NAME')
        if not isinstance(cls.collection_name, str):  # Check if it's a string
            raise TypeError("Collection name must be a string")
        return cls()

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.DB_URL)
        self.db = self.client[self.DB_NAME]

    def close_spider(self, spider):
        self.client.close()

    def process_item(self, item, spider):
        print(f"Collection name type: {type(self.collection_name)}")  # Check the type
        self.db[self.collection_name].insert_one(ItemAdapter(item).asdict())
        return item


# Pipeline for crawler for wdr
class MongoDBPipeline2(object):
    @classmethod
    def from_crawler(cls, crawler):
        cls.DB_URL = crawler.settings.get('MONGO_DB_URI')
        cls.DB_NAME = crawler.settings.get('MONGO_DB_NAME')
        cls.collection_name = crawler.settings.get('MONGO_COLLECTION_NAME2')
        if not isinstance(cls.collection_name, str):  # Check if it's a string
            raise TypeError("Collection name must be a string")
        return cls()

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.DB_URL)
        self.db = self.client[self.DB_NAME]

    def close_spider(self, spider):
        self.client.close()

    def process_item(self, item, spider):
        print(f"Collection name type: {type(self.collection_name)}")  # Check the type
        self.db[self.collection_name].insert_one(ItemAdapter(item).asdict())
        return item
