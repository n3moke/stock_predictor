import asyncio
from datetime import datetime, timedelta 
from enum import Enum
from io import StringIO
import os
import re
import pandas as pd
from ollama import chat
from ollama import ChatResponse
from ollama import AsyncClient
from nicegui import ui
from loguru import logger # type: ignore

import yfinance as yf

from dataprovider import Dataprovider
from util.perf import measure_time


class LLM:
    def __init__(self):
        self.sentiment_prompt_text = '''You will be provided with press realeases about {company}. Please rate the press release on a scale of -1 (very negative) to 1 (very positive) where 0 indicates neutral sentiment. Try to analyse the financial implications and their impact on the share price. Report the scores in increments of 0.1.
Please only answer with the sentiment score and do not include any other word or explanation before or after the sentiment score.
Do not include the '+' symbol in the output if the score is positive, but do inlcude the '-' symbol if the score is negative.'''

        self.arima_prompt_text = '''Please do a Time series forecasting for the closing price of {companyName} shares ({companyStockName}) using the ARIMA model. Please compute the forecast.
The following is csv data with date and close price of the stock\n{stock}\n
The file contains the date and the closing price for the {companyName} share, i.e. {companyStockName}.
The column "Date" represents the date. It is in the following formart: 2022-01-03 is for example the 3rd of January of 2022.
The column "Close" represents the closing price of the stock of that day and is in USD.
Important! The data covers the period from {startDayStr} ({start_date}) to {endDayStr} ({end_date}).
If data for certain days are missing, for example weekends, exclude them.
Provide me with the forecast for the {forcastRangeString}, excluding weekends and national holidays.
Just anwser with the forcast in a .csv format with "date" and "prediction" column.
Important! I want you to compute the forecast. I don't want python code!!!
Constraint: I only want {forcastDayRange} days of forecast after {end_date}. So your forcast should not contain more than {forcastDayRange} rows of data.'''
        self.combined_prompt_text = '''You are given two CSV strings with information about {companyName} and its stock {companyStockName}. Each contains date-indexed data:
First CSV: Has two columns, date and close price for the {companyName} share, i.e. {companyStockName}.
The column "Date" represents the date. It is in the following formart: 2022-01-03 is for example the 3rd of January of 2022.
The column "Close" represents the closing price of the stock of that day and is in USD.
Second CSV: Has two columns, date and sentiment_score.
The column "pubDate" represents the date. It is in the following formart: 2022-01-03 is for example the 3rd of January of 2022.
The column "Sentiment Score" represents the sentiment score of the news article for the date. It ranges from -1 (very negative) to 1 (very positive) where 0 indicates neutral sentiment

Important! The data covers the period from {startDayStr} ({start_date}) to {endDayStr} ({end_date}).
If data for certain days are missing in the first csv, for example weekends, exclude them.
If date is missing for certain dates in the second csv, there are simply no news articles for that specific day.

Your task is to compute a ARIMA forecast with the given data of the first csv. 
Your task is to combine these datasets by matching on the date column, and then compute a forecast ONLY! for the {forcastRangeString}, excluding weekends and national holidays. So your forcast should not contain more than {forcastRangeString} rows of data!
Use the sentiment score as an influencing factor to improve or refine the forecast.

First CSV String (historical stock data):
{sentiment_csv}
Second CSV (Sentiment):
{stock_csv}

Return the result of the ARIMA forecast of the closing price as a CSV string with two columns: date, prediction.
```csv
date,prediction
YYYY-MM-DD,###
YYYY-MM-DD,###
YYYY-MM-DD,###
```

Replace YYYY-MM-DD with forecasted dates and ### with the numeric predictions of the closing price.
The CSV must be inside the ```csv ``` block.
Do not include any additional text, explanations, code, or apologetic language
Constraint: I only want {forcastDayRange} days of forecast after {end_date}. So your forcast should only contain {forcastDayRange} rows of data. Also consider some months only contain never contain more than 31 days.
Important! I want you to compute the forecast. I don't want python code!!!

'''
 
        self.response_text: str = ''
        self.sentiment_response = pd.DataFrame({'pubDate': [''],'summary': [''],'Sentiment Score': [0]})
        self.dp = Dataprovider()
        self.isSentimentRunning: bool = False
        self.sentimentProgess:float = 0
        self.current_sentiment_number:int = 0
        self.numberOfSentiments:int = 0 
        self.arima_response = pd.DataFrame({'date': [''],'prediction': ['']})
        self.isArimaRunning:bool = False
        self.combined_response = pd.DataFrame({'date': [''],'prediction': ['']})
        self.combined_reasoning: str = ''


    class LLM_TYPE(Enum):    
        GEMMA3:str = 'gemma3'
        GEMMA3_27B:str = 'gemma3:27b'
        DEEPSEEK:str = 'deepseek-r1'
        DEEPSEEK70B:str = 'deepseek-r1:70b'
        GPT: str = 'gpt-oss:20b'

    async def _ask_llm_sentiment(self, llm_type: LLM_TYPE, input_content: str, system_prompt:str) ->  str | tuple[str, str]:
        logger.debug(f"LLM Type: {llm_type}")
        logger.debug(f"Input content \n{input_content} \n\n  prompt \n{system_prompt} \n\n")
        response: ChatResponse = await AsyncClient().chat(model=llm_type.value, messages=[
            {'role' : 'system', 'content' : system_prompt},
            {'role': 'user','content': input_content}
        ])
        response_text = response['message']['content']
        logger.debug(response_text)
        if(self.isSentimentRunning):
            #because of async.gather must use self.current_sentiment_number here for calculation
            self.current_sentiment_number += 1
            self.sentimentProgess = round(self.current_sentiment_number / self.numberOfSentiments,2)
            logger.debug(f"Sentiment Progress: {self.sentimentProgess}")
            logger.debug(f"current_sentiment_number: {self.current_sentiment_number}")
            logger.debug(f"numberOfSentiments: {self.numberOfSentiments}")
        if(llm_type in (self.LLM_TYPE.DEEPSEEK ,self.LLM_TYPE.DEEPSEEK70B)) :
            # Extract everything inside <think>...</think> - this is the Deep Think
            think_texts = re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
            # Join extracted sections (optional, if multiple <think> sections exist)
            think_texts = "\n\n".join(think_texts).strip()
            # Exclude the Deep Think, and return the response
            response_text= re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            logger.debug(response_text)
        self.response_text = response_text
        # Return either the context, or a tuple with the context and deep think if LLM.TYPE is DEEPSEEK or DEEPSEEK70B
        return response_text.rstrip() if not (llm_type in  (self.LLM_TYPE.DEEPSEEK ,self.LLM_TYPE.DEEPSEEK70B)) else (response_text, think_texts)


    async def ask_llm_prompt(self, llm_type: LLM_TYPE, system_prompt: str) ->  str | tuple[str, str]:
        logger.debug(f"LLM Type: {llm_type}")
        self.isArimaRunning = True
        response: ChatResponse = await AsyncClient().chat(model=llm_type.value, messages=[
            {'role': 'user','content': system_prompt}
        ])
        response_text = response['message']['content']
        self.isArimaRunning = False
        logger.debug(response_text)
        if(llm_type in (self.LLM_TYPE.DEEPSEEK ,self.LLM_TYPE.DEEPSEEK70B)) :
            think_texts = re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
            think_texts = "\n\n".join(think_texts).strip()
            response_text= re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            logger.debug(response_text)
        # Return either the context, or a tuple with the context and deep think if LLM.TYPE is DEEPSEEK or DEEPSEEK70B
        return response_text if not (llm_type in  (self.LLM_TYPE.DEEPSEEK ,self.LLM_TYPE.DEEPSEEK70B)) else (response_text, think_texts)
    
    async def _process_articles(self,llm_type:LLM_TYPE, news:pd.DataFrame, company: str):
        sent_prompt = self.sentiment_prompt_text.format(company=company)
        coroutines = [self._ask_llm_sentiment(llm_type, article, sent_prompt) for article in news['summary']]
        sentiments = await asyncio.gather(*coroutines)
        logger.debug(type(sentiments))
        news['Sentiment Score'] = sentiments
        self.sentiment_response = news
        self._resetProgessIndicators()
        return news

    def _resetProgessIndicators(self):
        self.isSentimentRunning = False
        self.sentimentProgess = 0
        self.current_sentiment_number = 0
        self.numberOfSentiments = 0 
        self.isArimaRunning = False

    @measure_time(write_log=True)
    async def get_sentiment_respone(self,llm_type: LLM_TYPE, stocks:list[str],start_date: datetime, end_date: datetime,newsType: Dataprovider.NEWSTYPE) -> pd.DataFrame:
        if not stocks:
            logger.warning("No stock was selected.")
            ui.notify('Please a stock first.')
            return self.sentiment_response
        else:
            stock = stocks[0]['stock']
            stockName = stocks[0]['name']
            logger.info(f"starting sentiment analysis of stock {stock}", )
            self.isSentimentRunning = True
            logger.debug(stock)
            logger.debug(f"From {start_date} to {end_date}")
            news = self.dp.fetch_news_single_ticker(stock, 200,start_date, end_date, newsType)
            self.numberOfSentiments = len(news)
            logger.debug(f"Number of news articles: {self.numberOfSentiments}")
            news = self.dp.prepare_news_sentiment(news)
            sentiment_analysis = await self._process_articles(llm_type, news, stockName)
            # Convert to datetime
            sentiment_analysis['pubDate'] = pd.to_datetime(sentiment_analysis['pubDate'])

            # Strip time, keep only date
            sentiment_analysis['pubDate'] = sentiment_analysis['pubDate'].dt.date
            logger.info(f"printing sentiment analysis result \n {sentiment_analysis}")
            self.save_response_as_csv(stock,sentiment_analysis,'sentiment', llm_type)
            return sentiment_analysis
    
    @measure_time(write_log=True)
    async def get_arima_response(self,llm_type: LLM_TYPE, stocks:list[str],start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if not stocks:
            logger.warning("No stock was selected.")
            ui.notify('Please a stock first.')
            return self.arima_response
        else:
            self.isArimaRunning = True
            stock = stocks[0]['stock']
            logger.debug(stock)
            logger.debug(f"From {start_date} to {end_date}")

            stockData = self.dp.fetch_historical_stock_data(stock, end_date, start_date)
            stockData = self.dp.prepare_stock_data_arima(stockData)

            prompt = self.format_arima_prompt(stockData,start_date,end_date,10,stocks)
            logger.debug(f"prompt: \n {prompt}")
            response = await self.ask_llm_prompt(llm_type,prompt)
            self._resetProgessIndicators()
            logger.info(response)
            arima_response = self._process_and_set_arima_response(response)
            self.save_response_as_csv(stock,arima_response,'arima', llm_type)
            return response

    @measure_time(write_log=True)   
    async def get_combined_response(self,llm_type: LLM_TYPE, stocks:list[str],start_date: datetime, end_date: datetime, newsType: Dataprovider.NEWSTYPE) -> pd.DataFrame:
        if not stocks:
            logger.warning("No stock was selected.")
            ui.notify('Please a stock first.')
            return self.arima_response
        else:
            self.isSentimentRunning = True
            stock = stocks[0]['stock']
            stockName = stocks[0]['name']

            #First do sentiment analysis
            news = self.dp.fetch_news_single_ticker(stock, 200,start_date, end_date, newsType)
            self.numberOfSentiments = len(news)
            logger.debug(f"Number of news articles: {self.numberOfSentiments}")
            news = self.dp.prepare_news_sentiment(news)
            sentiment_analysis = await self._process_articles(llm_type, news, stockName)
            # Convert to datetime
            sentiment_analysis['pubDate'] = pd.to_datetime(sentiment_analysis['pubDate'])

            # Strip time, keep only date
            sentiment_analysis['pubDate'] = sentiment_analysis['pubDate'].dt.date
            logger.info(f"Sentiment Result for combined forecast: \n {sentiment_analysis}")
            sentiment_csv = self.dp.prepare_sentiment_combined_forecast(sentiment_analysis)

            #now get historical stock data
            logger.debug(stock)
            logger.debug(f"From {start_date} to {end_date}")

            stockData = self.dp.fetch_historical_stock_data(stock, end_date, start_date)
            stock_csv = self.dp.prepare_stock_data_arima(stockData)

            #prepare prompt
            prompt = self.format_combined_prompt(sentiment_csv,stock_csv,start_date,end_date,10,stocks)
            logger.debug(f"formatted prompt: \n {prompt}")            
            response = await self.ask_llm_prompt(llm_type,prompt)
            
            logger.info(f"combined response: {response} \n##########################")
            processed_response =  self._process_and_set_combined_response(response)
            #save response as csv
            self.save_response_as_csv(stock,processed_response, 'combined',llm_type)
            self._resetProgessIndicators()
            return response

    def _process_and_set_arima_response(self, response:str) -> pd.DataFrame:
        pattern = r"^date,prediction(?:\r?\n\d{4}-\d{2}-\d{2},\d+(\.\d+)?)+"
        match = re.search(pattern, response, re.MULTILINE)
        logger.warning(match)
        if match:
            extracted_csv = match.group(0)
            logger.debug(f"Regex result: \n {extracted_csv}")
            arima_result = pd.read_csv(StringIO(extracted_csv))
            logger.debug(arima_result)
            self.arima_response = arima_result
            return arima_result
        else:
            return None
            

    def _process_and_set_combined_response(self, response) -> pd.DataFrame:
            # prediction_pattern = r"```csv([\s\S]*?)(?=```|###|@@@|[a-zA-Z])"
            # prediction_pattern = r"```csv([\s\S]*?)(?=```|###|@@@|[a-zA-Z])"

            prediction_pattern = r"```csv([\s\S]*?)```"
            match_sentiment = re.search(prediction_pattern, response)
            extracted_csv = match_sentiment.group(1) if match_sentiment else None
            logger.info(f"extracted csv: \n{extracted_csv}")
            #extract text between ### and ### or '''
            # reasoning_pattern = r"@@@([\s\S]*?)(?=```|###|@@@)"
            # match_reasoning = re.search(reasoning_pattern, response)
            # extracted_reasoning = match_reasoning.group(1) if match_reasoning else None
            combined_result = pd.read_csv(StringIO(extracted_csv))
            self.combined_response = combined_result
            # self.combined_reasoning = extracted_reasoning
            logger.info(combined_result)
            return combined_result

    def create_sentiment_table(self,data: pd.DataFrame) -> ui.table:
        data.columns = [f"{col}" for col in data.columns]
        table_data = data.to_dict('records')
        columns = [{'name': col, 'label': col, 'field': col} for col in data.columns]
        logger.debug(f"sentiment columns: {columns}")
        return ui.table(columns=columns, rows=table_data, row_key='Date',pagination={'rowsPerPage': 10})


    def get_sentiment_analysis_results(self,llm_tpye: LLM_TYPE, stocks:list[str]):
        stock = stocks[0]['stock']
        stockName = stocks[0]['name']
        news = self.dp.fetch_news_single_ticker(stock, 10,datetime(2020, 5, 17), datetime(2020, 5, 17), self.dp.NEWSTYPE.PRESS)
        news = self.dp.prepare_news_sentiment(news)
        sentiment_analysis = self._process_articles(llm_tpye, news, stockName)
        logger.info(sentiment_analysis)
        return sentiment_analysis
    
    def format_arima_prompt(self, csv: str,start_date:datetime, end_date:datetime, forcastrange: int, company : list[str]) -> str:
        daystring_format = "%d %B %Y"
        day_format = "%Y-%m-%d"
        forecast_Range_String = self.generate_forecastRange_string(forcastrange, end_date)
        prompt = self.arima_prompt_text.format(companyName=company[0]['stock'], companyStockName=company[0]['name'], stock=csv,start_date=start_date.strftime(day_format) ,end_date=end_date.strftime(day_format), startDayStr=start_date.strftime(daystring_format), endDayStr=end_date.strftime(daystring_format),forcastRangeString=forecast_Range_String,forcastDayRange=forcastrange)
        return prompt
    
    def format_combined_prompt(self, stock_csv: str,sentiment_csv :str,start_date:datetime, end_date:datetime, forcastrange: int, company : list[str]) -> str:
        daystring_format = "%d %B %Y"
        day_format = "%Y-%m-%d"
        forecast_Range_String = self.generate_forecastRange_string(forcastrange, end_date)
        prompt = self.combined_prompt_text.format(companyName=company[0]['stock'], companyStockName=company[0]['name'],sentiment_csv=sentiment_csv, stock_csv=stock_csv, start_date=start_date.strftime(day_format), end_date=end_date.strftime(day_format), startDayStr=start_date.strftime(daystring_format), endDayStr=end_date.strftime(daystring_format),forcastRangeString=forecast_Range_String, forcastDayRange=forcastrange)
        return prompt
    
    def generate_forecastRange_string(self,forcastrange:int, end_date:datetime):
        #example string  2025-08-16 to 2025-08-18
        end_date_plusOne = end_date + timedelta(days=1)
        end_date_total = end_date + timedelta(days=forcastrange)
        day_format = "%Y-%m-%d"
        forcastrangeString = f"{end_date_plusOne.strftime(day_format)} to {end_date_total.strftime(day_format)}"
        logger.debug(f"forcastrangeString: {forcastrangeString}")
        return forcastrangeString
    
    def save_response_as_csv(self, stock :str , data: pd.DataFrame, responseType: str, llm_type: LLM_TYPE):
        dir = 'results'
        os.makedirs(dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%d-%m-%d_%H_%M_%S')
        llm_string = llm_type.value.replace(":","_")
        filename = os.path.join(dir, f"{timestamp}_log_{responseType}_{stock}_{llm_string}.csv")
        logger.warning(f"data for csv : \n {data}" )
        data.to_csv(filename,index=False)

if __name__ == '__main__': 

    dp = Dataprovider()
    dp.__call__(['AAPL'])

    llm = LLM()
    stock_list = [{'name': 'Amazon', 'stock': 'AMZN'}]
    ##########################################
    #########       Sentiment   ##############
    ##########################################

    #fetch news for Amazon
    news = dp.fetch_news_single_ticker('AMZN', 10,datetime(2020, 5, 17), datetime(2020, 5, 17), dp.NEWSTYPE.PRESS)
    news = dp.prepare_news_sentiment(news)

    system_prompt_sentiment = '''You will be provided with press realeases about Apple Inc. Please rate the press release on a 
    scale of -1 (very negative) to 1 (very positive) where 0 indicates neutral sentiment. Report the scores in increments of 0.1.
    Please only answer with the sentiment score and do not include any other word or explanation before or after the sentiment score.
    Do not include the '+' symbol in the output if the score is positive, but do inlcude the '-' symbol if the score is negative.
    Try to analyse the financial implications and their impact on the share price.
    '''
    # news[['Sentiment Score']] = news['summary'].apply(lambda article : LLM.ask_llm_sentiment(llm.LLM_TYPE.GEMMA3,article, system_prompt_sentiment)).apply(pd.Series)
    resp_gemma = asyncio.run(llm.get_sentiment_respone(llm.LLM_TYPE.GEMMA3,stock_list))

    # resp_deep = asyncio.run(llm.get_sentiment_respone(llm.LLM_TYPE.DEEPSEEK,stock_list))
    # resp_gpt = asyncio.run(llm.get_sentiment_respone(llm.LLM_TYPE.GPT,stock_list))

    logger.info('GEMMA3 \n', resp_gemma, '\n\n')
    # logger.info('DEEPSEEK \n', resp_deep, '\n\n')
    # logger.info('GPT \n', resp_gpt, '\n\n')

    ##########################################
    ############     Arima    ################
    ##########################################

    #fetch stock data for Apple
    # end_date = datetime(2025,1,31)
    # start_date = datetime(2025,1,1)
    # stock = dp.fetch_historical_stock_data('AAPL', end_date, start_date)
    # stock = dp.prepare_stock_data_arima(stock)
    # system_prompt_arima = f'''Please do a Time series forecasting for the closing price of Apple shares (AAPL) using the ARIMA model. Please compute the forecast.
    # The following is csv data with date and close price of the stock
    # \n{stock}\n
    
    # The file contains the date and the closing price for the AAPL share, i.e. Apple. 
    # The column "Date" represents the date. It is in the following formart: 2022-01-03 is for example the 3rd of January of 2022.
    # The column "Close" represents the closing price of the stock of that day and is in USD.
    # Important! The data covers the period from 3 January 2025 (2025-01-03) to 30 January 2025 (2025-01-30).
    # If data for certain days are missing, for example weekends, exclude them.
    # Provide me with the forecast for the first 10 days of February 2025, excluding weekends and national holidays.

    # Just anwser with the forcast in a .csv format with "date" and "prediction" column
    # '''

    # response = asyncio.run(llm.ask_llm_prompt(llm.LLM_TYPE.GEMMA3,system_prompt_arima,False))
    # logger.info(response)