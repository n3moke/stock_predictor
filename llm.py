import asyncio
from datetime import datetime 
from enum import Enum
from io import StringIO
import re
import pandas as pd
from ollama import chat
from ollama import ChatResponse
from ollama import AsyncClient
from nicegui import ui

import yfinance as yf

from dataprovider import Dataprovider

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
Provide me with the forecast for the next {forcastRangeString} after {end_date}, excluding weekends and national holidays.
Just anwser with the forcast in a .csv format with "date" and "prediction" column.
Important! I want you to compute the forecast. I don't want python code!!!'''
        self.combined_prompt = '''You are given two csv files 
'''
        # Provide me with the forecast for the first 10 days of February 2025
        self.response_text = ''
        self.sentiment_response = pd.DataFrame({'pubDate': [''],'summary': [''],'Sentiment Score': [0]})
        self.dp = Dataprovider()
        self.isSentimentRunning = False
        self.sentimentProgess = 0
        self.current_sentiment_number = 0
        self.numberOfSentiments = 0 
        self.arima_response = pd.DataFrame({'date': [''],'prediction': ['']})
        self.isArimaRunning = False


    class LLM_TYPE(Enum):
        # def __str__(self):
        #     return str(self.value)
    
        GEMMA3:str = 'gemma3'
        DEEPSEEK:str = 'deepseek-r1'
        DEEPSEEK70B:str = 'deepseek-r1:70b'
        GPT: str = 'gpt-oss:20b'

    async def _ask_llm_sentiment(self, llm_type: LLM_TYPE, input_content: str, system_prompt:str , print_log : bool= True) ->  str | tuple[str, str]:
        if print_log : print(input_content, '\n\n', system_prompt, '\n\n')
        response: ChatResponse = await AsyncClient().chat(model=llm_type.value, messages=[
            {'role' : 'system', 'content' : system_prompt},
            {'role': 'user','content': input_content}
        ])
        response_text = response['message']['content']
        if print_log: print(response_text)
        if(self.isSentimentRunning):
            #because of async.gather must use self.current_sentiment_number here for calculation
            self.current_sentiment_number += 1
            self.sentimentProgess = round(self.current_sentiment_number / self.numberOfSentiments,2)
            print("Sentiment Progress: ", self.sentimentProgess)
            print("current_sentiment_number: ", self.current_sentiment_number)
            print("numberOfSentiments: ",self.numberOfSentiments )
        if(llm_type in (self.LLM_TYPE.DEEPSEEK ,self.LLM_TYPE.DEEPSEEK70B)) :
            # Extract everything inside <think>...</think> - this is the Deep Think
            think_texts = re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
            # Join extracted sections (optional, if multiple <think> sections exist)
            think_texts = "\n\n".join(think_texts).strip()
            # Exclude the Deep Think, and return the response
            response_text= re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            print(response_text)
        self.response_text = response_text
        # Return either the context, or a tuple with the context and deep think if LLM.TYPE is DEEPSEEK or DEEPSEEK70B
        return response_text.rstrip() if not (llm_type in  (self.LLM_TYPE.DEEPSEEK ,self.LLM_TYPE.DEEPSEEK70B)) else (response_text, think_texts)


    async def ask_llm_csv_in_prompt(self, llm_type: LLM_TYPE, system_prompt: str, print_log : bool= True) ->  str | tuple[str, str]:
        self.isArimaRunning = True
        if print_log : print('\n\n', system_prompt, '\n\n')
        response: ChatResponse = await AsyncClient().chat(model=llm_type.value, messages=[
            {'role': 'user','content': system_prompt}
        ])
        response_text = response['message']['content']
        self.isArimaRunning = False
        if print_log: print(response_text)
        if(llm_type in (self.LLM_TYPE.DEEPSEEK ,self.LLM_TYPE.DEEPSEEK70B)) :
            think_texts = re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
            think_texts = "\n\n".join(think_texts).strip()
            response_text= re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            print(response_text)
        # Return either the context, or a tuple with the context and deep think if LLM.TYPE is DEEPSEEK or DEEPSEEK70B
        return response_text if not (llm_type in  (self.LLM_TYPE.DEEPSEEK ,self.LLM_TYPE.DEEPSEEK70B)) else (response_text, think_texts)
    
    async def _process_articles(self,llm_type:LLM_TYPE, news:pd.DataFrame, company: str,print_log = True):
        sent_prompt = self.sentiment_prompt_text.format(company=company)
        coroutines = [self._ask_llm_sentiment(llm_type, article, sent_prompt,print_log) for article in news['summary']]
        sentiments = await asyncio.gather(*coroutines)
        print(type(sentiments))
        news['Sentiment Score'] = sentiments
        self.sentiment_response = news
        self._resetSentiment()
        
        return news

    def _resetSentiment(self):
        self.isSentimentRunning = False
        self.sentimentProgess = 0
        self.current_sentiment_number = 0
        self.numberOfSentiments = 0 

    # async def get_sentiment_respone(self,llm_tpye: LLM_TYPE, aktien:list[str],print_log = True) -> pd.DataFrame:
    #     print(aktien)
    #     print(aktien[0]['aktie'])
    #     news = self.dp.fetch_news_single_ticker(aktien[0]['aktie'], 10,datetime(2020, 5, 17), datetime(2020, 5, 17), self.dp.NEWSTYPE.PRESS)
    #     news = self.dp.prepare_news_sentiment(news)
    #     sentiment_analysis = await self._process_articles(llm_tpye, news, aktien[0]['name'],print_log)
    #     print(sentiment_analysis)
    #     print(type(sentiment_analysis))
    #     return self.create_sentiment_table(sentiment_analysis)
    
    async def get_sentiment_respone(self,llm_tpye: LLM_TYPE, aktien:list[str],start_date: datetime, end_date: datetime,print_log = True) -> pd.DataFrame:
        if not aktien:
            ui.notify('Please a stock first.')
            return self.sentiment_response
        else:
            self.isSentimentRunning = True
            print(aktien)
            print(aktien[0]['aktie'])
            print(start_date, '  ', end_date)
            news = self.dp.fetch_news_single_ticker(aktien[0]['aktie'], 200,start_date, end_date, self.dp.NEWSTYPE.PRESS)
            self.numberOfSentiments = len(news)
            print(self.numberOfSentiments)
            news = self.dp.prepare_news_sentiment(news)
            sentiment_analysis = await self._process_articles(llm_tpye, news, aktien[0]['name'],print_log)
            # Convert to datetime
            sentiment_analysis['pubDate'] = pd.to_datetime(sentiment_analysis['pubDate'])

            # Strip time, keep only date
            sentiment_analysis['pubDate'] = sentiment_analysis['pubDate'].dt.date
            print(sentiment_analysis)
            print(type(sentiment_analysis))
            return sentiment_analysis
        #return self.create_sentiment_table(sentiment_analysis)
    

    async def get_arima_response(self,llm_type: LLM_TYPE, aktien:list[str],start_date: datetime, end_date: datetime,print_log = True) -> pd.DataFrame:
        if not aktien:
            ui.notify('Please a stock first.')
            return self.arima_response
        else:
            self.isArimaRunning = True
            print(aktien)
            print(aktien[0]['aktie'])
            print(start_date, '  ', end_date)

            stock = self.dp.fetch_historical_stock_data(aktien[0]['aktie'], end_date, start_date)
            stock = self.dp.prepare_stock_data_arima(stock)

            prompt = self.format_arima_prompt(stock,start_date,end_date,10,aktien)
            print(prompt)
            response = await self.ask_llm_csv_in_prompt(llm_type,prompt,False)
            self.isArimaRunning = False
            print(response)
            self._process_and_set_arima_response(response)
            return response

    def _process_and_set_arima_response(self, csv:str):
            pattern = r"```csv([\s\S]*?)```"
            match = re.search(pattern, csv)
            extracted_csv = match.group(1) if match else None
            print('Regex result: ', extracted_csv)
            
            self.arima_response = pd.read_csv(StringIO(extracted_csv))
            print(self.arima_response)


    def create_sentiment_table(self,data: pd.DataFrame) -> ui.table:
        data.columns = [f"{col}" for col in data.columns]
        print(data.columns)
            #df.rename(columns={ df.columns[0]: "Close" }, inplace = True)
            #df.columns = df.columns.to_flat_index()
            #df.insert(0, 'Date', df.index.strftime('%Y-%m-%d'))
            #df = df.rename(columns={ df.columns[1]: "Close" }, inplace = True)
            #df['Close'] = df['Close'].round(3)
            #df.round(2)
            # DataFrame als Liste von Dictionaries für NiceGUI (Table) konvertieren
        table_data = data.to_dict('records')

            # Spaltennamen für NiceGUI-Table
        columns = [{'name': col, 'label': col, 'field': col} for col in data.columns]
        print(columns)
            #columns = [['Date', 'Close']]

            # NiceGUI Table anzeigen
        return ui.table(columns=columns, rows=table_data, row_key='Date',pagination={'rowsPerPage': 10})


    def get_sentiment_analysis_results(self,llm_tpye: LLM_TYPE, aktien:list[str],print_log = True):
        print(aktien)
        print(aktien[0]['aktie'])
        news = self.dp.fetch_news_single_ticker(aktien[0]['aktie'], 10,datetime(2020, 5, 17), datetime(2020, 5, 17), self.dp.NEWSTYPE.PRESS)
        news = self.dp.prepare_news_sentiment(news)
        sentiment_analysis = self._process_articles(llm_tpye, news, aktien[0]['name'],print_log)
        print(sentiment_analysis)
        print(type(sentiment_analysis))
        return sentiment_analysis
    
    def format_arima_prompt(self, csv: str,start_date:datetime, end_date:datetime, forcastrange: int, company : list[str]) -> str:
        daystring_format = "%d %B %Y"
        day_format = "%Y-%m-%d"
        prompt = self.arima_prompt_text.format(companyName=company[0]['aktie'], companyStockName=company[0]['name'], stock=csv,start_date=start_date.strftime(day_format) ,end_date=end_date.strftime(day_format), startDayStr=start_date.strftime(daystring_format), endDayStr=end_date.strftime(daystring_format),forcastRangeString=forcastrange)
        return prompt

if __name__ == '__main__': 

    dp = Dataprovider()
    dp.__call__(['AAPL'])

    llm = LLM()
    stock_list = [{'name': 'Amazon', 'aktie': 'AMZN'}]
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

    print('GEMMA3 \n', resp_gemma, '\n\n')
    # print('DEEPSEEK \n', resp_deep, '\n\n')
    # print('GPT \n', resp_gpt, '\n\n')

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

    # response = asyncio.run(llm.ask_llm_csv_in_prompt(llm.LLM_TYPE.GEMMA3,system_prompt_arima,False))
    # print(response)