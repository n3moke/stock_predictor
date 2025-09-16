from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataprovider import Dataprovider

class Settings:
    def __init__(self):
        self.timeframe: int = 1
        self.end_date_str = '2025-08-31'
        self.arima = True
        self.lstm = True
        self.aktien = []
        self.aktien_list = []
        self.end_date = datetime.strptime(self.end_date_str, "%Y-%m-%d")
        self.start_date = self.end_date - relativedelta(months=self.timeframe)
        self.forecastMethod: int = 0
        self.useArima: bool = True
        self.useSentiment: bool = False
        self.useCombined: bool = False
        self.newsTypeInteger: int = 0
        self.newsType: Dataprovider.NEWSTYPE = Dataprovider.NEWSTYPE.PRESS

    def __call__(self, aktien):
        self.aktien = aktien
    
    def update_aktien(self,aktien):
        self.aktien = aktien
        print(self.aktien)

    def update_dates(self):
        self.end_date = datetime.strptime(self.end_date_str, "%Y-%m-%d")
        self.start_date = self.end_date - relativedelta(months=self.timeframe)
        print('end_date_str: ' , self.end_date_str,'end_date: ',self.end_date,'start_date: ', self.start_date,'timeframe: ', self.timeframe)

    def update_forecast_method(self):
        print(self.forecastMethod)
        self.useArima = (self.forecastMethod == 0)
        self.useSentiment = (self.forecastMethod == 1)
        self.useCombined = (self.forecastMethod == 2)

    def update_newsType(self): 
        if(self.newsTypeInteger == 0):
            self.newsType = Dataprovider.NEWSTYPE.PRESS
        elif(self.newsTypeInteger == 1):
            self.newsType = Dataprovider.NEWSTYPE.NEWS
        elif(self.newsTypeInteger == 2):
            self.newsType = Dataprovider.NEWSTYPE.ALL