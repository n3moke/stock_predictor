from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataprovider import Dataprovider
from llm import LLM
from loguru import logger # type: ignore

class Settings:
    def __init__(self):
        self.timeframe: int = 1
        self.end_date_str = '2025-08-31'
        self.stocks = []
        self.end_date = datetime.strptime(self.end_date_str, "%Y-%m-%d")
        self.start_date = self.end_date - relativedelta(months=self.timeframe)
        self.forecastMethod: int = 0
        self.useArima: bool = True
        self.useSentiment: bool = False
        self.useCombined: bool = False
        self.newsTypeInteger: int = 0
        self.newsType: Dataprovider.NEWSTYPE = Dataprovider.NEWSTYPE.PRESS
        self.llmTypeInteger: int = 0
        self.llmType: LLM.LLM_TYPE = LLM.LLM_TYPE.GEMMA3
    
    def update_stocks(self,stocks):
        self.stocks = stocks
        logger.info(self.stocks)

    def update_dates(self):
        self.end_date = datetime.strptime(self.end_date_str, "%Y-%m-%d")
        self.start_date = self.end_date - relativedelta(months=self.timeframe)
        logger.debug(f"end_date_str: {self.end_date_str} end_date: {self.end_date} start_date: {self.start_date} timeframe: {self.timeframe}")

    def update_forecast_method(self):
        logger.info(f"forcastMethod: {self.forecastMethod}")
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

    def update_llmType(self): 
        if(self.llmTypeInteger == 0):
            self.llmType = LLM.LLM_TYPE.GEMMA3
        elif(self.llmTypeInteger == 1):
            self.llmType = LLM.LLM_TYPE.GEMMA3_27B
        elif(self.llmTypeInteger == 2):
            self.llmType = LLM.LLM_TYPE.DEEPSEEK
        elif(self.llmTypeInteger == 3):
            self.llmType = LLM.LLM_TYPE.DEEPSEEK70B
        elif(self.llmTypeInteger == 4):
            self.llmType = LLM.LLM_TYPE.GPT