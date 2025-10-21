# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class Sp500PerfItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    rank = scrapy.Field()
    company = scrapy.Field()
    symbol = scrapy.Field()
    ytd_return = scrapy.Field()
