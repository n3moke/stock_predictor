import asyncio
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from enum import Enum
import pandas as pd
import yfinance as yf
from nicegui import ui, background_tasks
import json
from loguru import logger # type: ignore

class Dataprovider:
    class NEWSTYPE(Enum):
        PRESS:str = 'press releases'
        ALL:str = 'all'
        NEWS:str = 'news'

    def __init__(self):
        self.stocks_list: list[str] = []
        self.stock_table_data: pd.DataFrame = pd.DataFrame({'Date': [''],'Close': ['']})
    def __call__(self, aktien:list[str]):
        self.aktien = aktien

    def load_stocks(self):
        with open('stocklist.json', 'r') as file:
            self.stocks_list = json.load(file)

    def fetch_historical_stock_data(self,aktien:str, end_date: datetime, start_date:datetime) -> pd.DataFrame:
        return yf.download(aktien, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))[['Close']]
    
    
    def fetch_news_single_ticker(self, aktie:str ,newscount:int, start_date: datetime,  end_date: datetime, news_type: NEWSTYPE ) -> list[dict]:
        ticker = yf.Ticker(aktie)
        logger.info(f"news type: {news_type.value}")
        news = ticker.get_news(count=newscount,tab=news_type.value)
        filtered_news:pd.DataFrame = self.filter_news_daterange(news, start_date,end_date)
        return filtered_news

    def filter_news_daterange(self, news: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        start = start_date.replace(tzinfo=timezone.utc)
        end = end_date.replace(tzinfo=timezone.utc)
        filtered_news = []
        for article in news:
            pubdate = datetime.fromisoformat(article['content']['pubDate'].replace('Z', '+00:00'))
            if (start <= pubdate <= end):
                filtered_news.append(article)
                logger.trace("added article")
        return filtered_news

    def prepare_news_sentiment(self, news: list[dict]) -> pd.DataFrame:
        # put only summary and pubdate into a pandas dataframe
        df = pd.DataFrame([
            { 'pubDate': article['content']['pubDate'], 'summary' : article['content']['summary']}
            for article in news
        ])
        logger.debug(df)
        return df

    def prepare_stock_data_arima(self, data: pd.DataFrame) -> str:
        #round column Close to 3 decimal places
        data['Close'] = data['Close'].round(3)
        data.columns = [f"{col}" for col in data.columns]
        data.columns = data.columns.to_flat_index()
        data.insert(0, 'Date', data.index.strftime('%Y-%m-%d'))
        data.columns = ['Date', 'Close']
        csv = data.to_csv(index=False)
        logger.debug(f"prepared csv: \n {csv}")
        return csv
    
    def prepare_sentiment_combined_forecast(self,data:pd.DataFrame) -> str:
        #drop summary column
        data.drop('summary', axis=1, inplace=True)
        csv = data.to_csv(index=False)
        logger.debug(f"prepared csv: \n {csv}")
        return csv


    def create_table_multiple_stocks(self, aktien:list[str]) -> ui.table:
        self.stocks_list = []
        if not aktien:
            logger.warning("No stock selected. Returning empty table")
            return ui.table()
        else:
            for aktie in aktien:
                self.stocks_list.append(aktie['aktie'])
            logger.debug(f"printing stocks_list: \n {self.stocks_list}")
            df = self.fetch_historical_stock_data(self.stocks_list)

            df.columns = [f"{col}" for col in df.columns]
            #df.columns = df.columns.to_flat_index()
            df.insert(0, 'Date', df.index.strftime('%Y-%m-%d'))
            df.round(2)
            # DataFrame als Liste von Dictionaries für NiceGUI (Table) konvertieren
            table_data = df.to_dict('records')

            # Spaltennamen für NiceGUI-Table
            columns = [{'name': col, 'label': col, 'field': col} for col in df.columns]

            # NiceGUI Table anzeigen
            return ui.table(columns=columns, rows=table_data, row_key='Date',pagination={'rowsPerPage': 10})
    
    def create_table_single_stock(self, aktien:list[str], end_date: datetime, start_date:datetime) -> ui.table:
        logger.info(f"End Date: {end_date} Start Date: {start_date}")
        self.stocks_list = []
        if not aktien:
            logger.warning("No stock selected. Returning empty table")
            return None
        else:
            for aktie in aktien:
                self.stocks_list.append(aktie['stock'])
            logger.debug(f"printing stocks_list: \n {self.stocks_list}")
            df = self.fetch_historical_stock_data(self.stocks_list[0], end_date, start_date)

            df.columns = [f"{col}" for col in df.columns]
            logger.trace(df.columns)
            df.rename(columns={ df.columns[0]: "Close" }, inplace = True)
            #df.columns = df.columns.to_flat_index()
            df.insert(0, 'Date', df.index.strftime('%Y-%m-%d'))
            #df = df.rename(columns={ df.columns[1]: "Close" }, inplace = True)
            df['Close'] = df['Close'].round(3)
            #df.round(2)
            # DataFrame als Liste von Dictionaries für NiceGUI (Table) konvertieren
            table_data = df.to_dict('records')

            # Spaltennamen für NiceGUI-Table
            columns = [{'name': col, 'label': col, 'field': col} for col in df.columns]
            logger.trace(columns)
            #columns = [['Date', 'Close']]

            # NiceGUI Table anzeigen
            return ui.table(columns=columns, rows=table_data, row_key='Date',pagination={'rowsPerPage': 10})

    def get_pandas_single_stock(self, aktien:list[str], end_date: datetime, start_date:datetime) -> pd.DataFrame:
        logger.info(f"End Date: {end_date} Start Date: {start_date}")
        self.stocks_list = []
        if not aktien:
            logger.warning("No stock selected. Returning empty table")
            return None
        else:
            for aktie in aktien:
                self.stocks_list.append(aktie['stock'])
            logger.debug(f"printing stocks_list: \n {self.stocks_list}")
            df = self.fetch_historical_stock_data(self.stocks_list[0], end_date, start_date)

            df.columns = [f"{col}" for col in df.columns]
            logger.trace(df.columns)
            df.rename(columns={ df.columns[0]: "Close" }, inplace = True)
            df.insert(0, 'Date', df.index.strftime('%Y-%m-%d'))
            df['Close'] = df['Close'].round(3)
            return df
        

    def get_pandas_single_stock_table(self, aktien:list[str], end_date: datetime, start_date:datetime) -> ui.table:
        logger.info(f"End Date: {end_date} Start Date: {start_date}")
        self.stocks_list = []
        if not aktien:
            logger.warning("No stock selected. Returning empty table")
            return None
        else:
            for aktie in aktien:
                self.stocks_list.append(aktie['stock'])
            logger.debug(f"printing stocks_list: \n {self.stocks_list}")
            df = self.fetch_historical_stock_data(self.stocks_list[0], end_date, start_date)

            df.columns = [f"{col}" for col in df.columns]
            logger.trace(df.columns)
            df.rename(columns={ df.columns[0]: "Close" }, inplace = True)
            df.insert(0, 'Date', df.index.strftime('%Y-%m-%d'))
            df['Close'] = df['Close'].round(3)
            logger.debug(df)
            return ui.table.from_pandas(df)

if __name__ == '__main__':
    dataprovider = Dataprovider()
    dataprovider.__call__(['AAPL'])
    
    #fetch news
    news = dataprovider.fetch_news_single_ticker('AAPL', 10,datetime.datetime(2020, 5, 17), datetime.datetime(2020, 5, 17), dataprovider.NEWSTYPE.PRESS)
    dataprovider.prepare_news_sentiment(news)

    #fetch stock data
    end_date = datetime(2025,1,31)
    start_date = datetime(2025,1,1)
    stock = dataprovider.fetch_historical_stock_data('AAPL', end_date,start_date)
    csv_arima = dataprovider.prepare_stock_data_arima(stock)

