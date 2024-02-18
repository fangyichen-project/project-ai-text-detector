
import scrapy
import re

from scrapy import Spider
from scrapy.selector import Selector
import subprocess
import sys
from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerProcess
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor
from scrapy.utils.log import configure_logging



from scrapy.utils.reactor import install_reactor


install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")

#%%
# Add a directory to the Python search path
sys.path.append('/Users/fangyichen/Project Text Detector的副本/dlf_spider/bremenspider')

# Now you can import modules from this directory
from items import CustomItem  # Importing the CustomItem class

# Write a spider that scrapes from the website of radiobremen
class BremenSpiderText(scrapy.Spider):
    name = "bremen_spider_text"
    allowed_domains = ["butenunbinnen.de"]

    custom_settings = {
        'DEPTH_LIMIT': 2,  # Limit the spider to 2 pages
        'DOWNLOAD_DELAY': 5,
        'ITEM_PIPELINES': {'bremenspider.pipelines.MongoDBPipeline':300}}
    
    def start_requests(self):
        base_url = "https://www.butenunbinnen.de/nachrichten/nachrichten738~_category-all_from-0_site-butenunbinnen_till-0_page-{}.html"
        # Assuming the pattern is from 1 to 100 for demonstration purposes
        for i in range(1, 100):
            url = base_url.format(i)
            yield scrapy.Request(url, callback=self.parse)


    def parse(self, response):
        links= response.css("a[class^='teaser-link']::attr(href)").getall()
        for link in links:
            if link.startswith('/nachrichten'):
                absolute_url = response.urljoin(link)
                yield scrapy.Request(absolute_url, callback=self.parse_text,meta={'url': absolute_url})

    def parse_text(self, response):
        # Extract the text content
        item = CustomItem()
        item['text'] = response.css('p:not([class])::text').getall()
        item['links'] = response.meta['url']
        
        extracted_text = ' '.join(item['text']).strip()
        if extracted_text:
            yield item
            
#%%
   
runner = CrawlerRunner(get_project_settings())
d = runner.crawl(BremenSpiderText)
d.addBoth(lambda _: reactor.stop())
reactor.run()  # the script will block here until the crawling is finished          


#%%
'''
def run_brm_spider():
    process = CrawlerProcess(get_project_settings())
    process.crawl(BremenSpiderText)
    try:
        process.start(stop_after_crawl=False)
        
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt (Ctrl+C)
        process.stop()



if __name__ == "__main__":
    run_brm_spider()
    
run_brm_spider()

'''