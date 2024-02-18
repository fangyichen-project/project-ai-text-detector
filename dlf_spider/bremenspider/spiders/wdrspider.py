'''
Spider to crawl text form wdr website.
'''
import scrapy
import urllib.parse
from scrapy.item import Item, Field
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

#%%
#Define Items for the pipeline
class CustomItem(Item):
    links = Field()
    text = Field()

#%%
#Define the spider
#Get the links from the website https://www1.wdr.de/abisz120.html
class SecondSpiderSpider(scrapy.Spider):
    name = "second_spider"
    allowed_domains = ["www1.wdr.de"]
    start_urls = ["https://www1.wdr.de/abisz120.html"]
    
    custom_settings = {
        'DEPTH_LIMIT': 0,
        'DOWNLOAD_DELAY': 5,
        'ITEM_PIPELINES':{'bremenspider.pipelines.MongoDBPipeline2':300},
        'MONGO_COLLECTION_NAME2': 'human_written_texts(wdr)' # Collection for SecondSpider
    }
    
#%%        
#Get the links from the website https://www1.wdr.de/abisz120.html
    def parse(self, response):
        links1 = response.css('ul.list a::attr(href)').getall()
        base_link = 'https://www1.wdr.de/'
        for link in links1:
            absolute_url = urllib.parse.urljoin(base_link, link)
            yield scrapy.Request(url=absolute_url, callback=self.link_parse)
            
#%%          
#Get the links from the links one layer deeper than the mentioned the website           
    def link_parse(self, response):
        links2 = response.css('div.teaser a::attr(href)').getall()
        base_link2 = 'https://www1.wdr.de/'
        for link in links2:
            absolute_url2 = urllib.parse.urljoin(base_link2, link)
            yield scrapy.Request(absolute_url2, callback=self.text_parse,
                                 meta={'url': absolute_url2})
#%%
#Get the text from the links
    def text_parse(self, response):
        item = CustomItem()
        item['text'] = response.xpath('//p[@class="text small"]/descendant-or-self::*/text()').getall()
        item['links'] = response.meta['url']
        extracted_text = ' '.join(item['text']).strip()
        if extracted_text:
            yield item
            
#%%
# Define a function to run the spider from main file.

def run_wdr_spider():
    process = CrawlerProcess(get_project_settings())
    process.crawl(SecondSpiderSpider)
    try:
        process.start(stop_after_crawl=False)
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt (Ctrl+C)
        process.stop()

if __name__ == "__main__":
    run_wdr_spider()
    
