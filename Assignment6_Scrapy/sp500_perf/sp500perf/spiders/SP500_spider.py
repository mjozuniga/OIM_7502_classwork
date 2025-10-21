import scrapy

class SP500Spider(scrapy.Spider):
    name = "sp500"
    allowed_domains = ["slickcharts.com"]
    start_urls = ["https://www.slickcharts.com/sp500/performance"]

    def parse(self, response):
        rows = response.xpath('//table[contains(@class,"table")]/tbody/tr')
        for r in rows:
            yield {
                "number": r.xpath('td[1]/text()').get(),
                "company": r.xpath('td[2]/a/text()').get(),
                "symbol": r.xpath('td[3]/text()').get(),
                "ytd_return": r.xpath('td[4]/text()').get(),
            }





