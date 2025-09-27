import asyncio
import datetime
from nicegui import ui, background_tasks
import pandas as pd
import yfinance as yf

from dataprovider import Dataprovider
from llm import LLM
from settings import Settings
from loguru import logger # type: ignore

settings = Settings()
llm = LLM()
dataprovider = Dataprovider()
dataprovider.load_stocks()
ui.label("Generative AI - stock prediction").classes('text-xl')
with ui.tabs().classes('w-full') as tabs:
    one = ui.tab('Selection')
    two = ui.tab('Data')
    three = ui.tab('Prompt')
    four = ui.tab('Prediction')
with ui.tab_panels(tabs, value=one).classes('w-full'):
    with ui.tab_panel(one):
        ui.label('Select period of historical data').classes('text-lg')
        ui.toggle({1:'1 month', 3:'3 months',6:' 6 months',12: '12 months',60: '5 years'},on_change=settings.update_dates).bind_value(settings,'timeframe')
        ui.label('Choose end date')
        with ui.input('Date').bind_value(settings, 'end_date_str') as date:
            with ui.menu().props('no-parent-event') as menu:
                with ui.date(value=settings.end_date_str, on_change=settings.update_dates).bind_value(settings,'end_date_str'):
                    with ui.row().classes('justify-end'):
                        ui.button('Close', on_click=menu.close).props('flat')
            with date.add_slot('append'):
                ui.icon('edit_calendar').on('click', menu.open).classes('cursor-pointer')
        ui.label('Stockselection').classes('text-lg')
        search_input = ui.input(placeholder="Search for stock").props('clearable')
        table = ui.table(
            columns=[
            {'name': 'name', 'label': 'CompanyName', 'field': 'name', 'sortable': True, 'align': 'left'},
            {'name': 'Stock', 'label': 'stockSymbol', 'field': 'stock', 'sortable': True}],
            rows=dataprovider.stocks_list,
            row_key='name',
            on_select=lambda e: settings.update_stocks(e.selection),
            pagination=10
        )
        search_input.bind_value_to(table, 'filter')
        table.set_selection('single')
        ui.button('GOTO data tab', on_click=lambda: tabs.set_value(two))
    with ui.tab_panel(two):
        stock_table_container = ui.column()
        with stock_table_container as stock_parent_container:
            stockdata_table :ui.table = dataprovider.get_pandas_single_stock_table(settings.stocks, settings.end_date, settings.start_date)

        ui.button('Refresh Stock Data', on_click=lambda _: set_and_update_table())
        goToTabThree =  ui.button('Goto prompt tab', on_click=lambda: tabs.set_value(three))
        # ui.switch(value=True, on_change=lambda e: goToTabThree.props(remove='disabled') if e.value else goToTabThree.props('disabled'))
        
                    
        def set_and_update_table():
            if not settings.stocks:
                ui.notify('Please select a stock first to retreive historical data!')
                logger.warning(settings.stocks)
            with stock_table_container as stock_parent_container:
                if not settings.stocks:
                    logger.warning("No stock was selected, while trying to load historical data")
                else:
                    logger.info('deleting children of stock_parent_container and add it again to update it...')
                    stock_parent_container.clear()                
                    df = dataprovider.get_pandas_single_stock(settings.stocks, settings.end_date, settings.start_date)
                    ui.table.from_pandas(df,row_key='Date',pagination={'rowsPerPage': 10},title=settings.stocks[0]['name'])
    with ui.tab_panel(three):
        ui.label('Choose LLM').classes('text-lg')
        ui.toggle({0:'GEMMA3',1:'GEMMA3_27B', 2:'DEEPSEEK',3: 'DEEPSEEK70B',4: 'GPT-OSS:20b'},on_change=settings.update_llmType).bind_value(settings,'llmTypeInteger')
        ui.label('Choose forecast method').classes('text-lg')
        ui.toggle({0:'ARIMA', 1:'Sentiment',2: 'ARIMA & Sentiment'},on_change=settings.update_forecast_method).bind_value(settings,'forecastMethod')
        with ui.card().bind_visibility(settings,'useArima').classes('w-full').props("autogrow") as arima_card:
            ui.textarea(label='Text', placeholder='start typing').bind_value(llm, 'arima_prompt_text').classes('w-full').props('autogrow input-style="min-height: 300px"')
            ui.button('Calculate Arima forecast', on_click=lambda:  do_arima_forecast()).tooltip('A arima is done with the given timeframe selected in the first tab. The accuracy befenits normally from a longer timeframe.')       
            ui.spinner('dots', size='lg', color='red').bind_visibility(llm, 'isArimaRunning')

        with ui.card().bind_visibility(settings,'useSentiment').classes('w-full').props("autogrow") as sentiment_card:
            ui.toggle({0:'Press Releases', 1:'News',2: 'All'},on_change=settings.update_newsType).bind_value(settings,'newsTypeInteger')
            ui.textarea(label='Text', placeholder='start typing').bind_value(llm, 'sentiment_prompt_text').classes('w-full').props('autogrow input-style="min-height: 300px"')

            ui.button('Do Sentiment Analysis', on_click=lambda:  do_sentiment_analysis()).tooltip('A sentiment analysis is done with press releases of the selected company and the given timeframe. It could be, that no press releases are available for the timeframe.')
        
            sentiment_progressbar = ui.linear_progress(value=0).props('instant-feedback').bind_visibility(llm,'isSentimentRunning').bind_value(llm, 'sentimentProgess')
            sentiment_progressbar.visible = False

        with ui.card().bind_visibility(settings,'useCombined').classes('w-full').props("autogrow") as combined_card:
            ui.toggle({0:'Press Releases', 1:'News',2: 'All'},on_change=settings.update_newsType).bind_value(settings,'newsTypeInteger')
            ui.textarea(label='Text', placeholder='start typing').bind_value(llm, 'combined_prompt_text').classes('w-full').props('autogrow input-style="min-height: 300px"')
            ui.button('Do combined forecast', on_click=lambda:  do_combined_forecast()).tooltip('A arima is done with the given timeframe selected in the first tab. In addition a sentiment analysis is done with press releases of the selected company and the given timeframe. The accuracy befenits normally from a longer timeframe.')       
            combined_progressbar = ui.linear_progress(value=0).props('instant-feedback').bind_visibility(llm,'isSentimentRunning').bind_value(llm, 'sentimentProgess')
            combined_progressbar.visible = False
            ui.spinner('dots', size='lg', color='red').bind_visibility(llm, 'isArimaRunning')
     
        def do_sentiment_analysis():
            if checkStockIsSelected(): background_tasks.create(llm.get_sentiment_respone(settings.llmType,settings.stocks,settings.start_date, settings.end_date, settings.newsType))

        def do_arima_forecast():
            if checkStockIsSelected(): background_tasks.create(llm.get_arima_response(settings.llmType,settings.stocks,settings.start_date, settings.end_date))

        def do_combined_forecast():
            if checkStockIsSelected(): background_tasks.create(llm.get_combined_response(settings.llmType,settings.stocks,settings.start_date, settings.end_date,settings.newsType))
        
        def checkStockIsSelected() -> bool :
            if not settings.stocks:
                logger.warning('No stock was selected while trying to start a sentiment analysis')
                ui.notify('Please select a in tab 1 stock first!')
                return False
            return True
            
    with ui.tab_panel(four):
        ui.label('Prediction').classes('text-lg')
        
        with ui.card().bind_visibility(settings,'useArima').classes('w-full') as arima_result_card:
            arima_table_container = ui.column()
            with arima_table_container as arima_parent_container:
                ui.table.from_pandas(llm.arima_response)
                
        with ui.card().bind_visibility(settings,'useSentiment').classes('w-full') as sentiment_result_card:
            sent_table_container = ui.column()
            with sent_table_container as sent_parent_container:
                ui.table.from_pandas(llm.sentiment_response)
        
        with ui.card().bind_visibility(settings,'useCombined').classes('w-full') as combined_result_card:
            combined_table_container = ui.column()
            with combined_table_container as combined_parent_container:
                ui.table.from_pandas(llm.combined_response)

        ui.button('Load Result', on_click=lambda _: display_predictions())

        def display_predictions():
            set_sentiment_table()
            set_arima_table()
            set_combined_table()
        
        def set_sentiment_table():
            with sent_table_container as sent_parent_container:
                sent_parent_container.clear()
                logger.info(f"Sentiment response: \n {llm.sentiment_response}")
                
                sent_table_title = f"Sentiment Analysis of press releases about {settings.stocks[0]['name']}"
                ui.table.from_pandas(llm.sentiment_response, title=sent_table_title, pagination={'rowsPerPage': 10}).classes('w-full')


        def set_arima_table():
            with arima_table_container as arima_parent_container:
                arima_parent_container.clear()
                arima_table_title = f"Arima stock forecast for {settings.stocks[0]['name']}"
                ui.table.from_pandas(llm.arima_response, title=arima_table_title, pagination={'rowsPerPage': 10}).classes('w-full')
        
        def set_combined_table():
             with combined_table_container as combined_parent_container:
                combined_parent_container.clear()
                logger.info(f"Combined response: \n {llm.combined_response}")
                
                sent_table_title = f"Combined Prediction (ARIMA and Sentiment Analysis) of {settings.stocks[0]['name']}"
                ui.table.from_pandas(llm.combined_response, title=sent_table_title, pagination={'rowsPerPage': 10}).classes('w-full')
                ui.label("Reasoning").bind_text_from(llm, 'combined_reasoning')


#ui.run(favicon='res/icon.png',dark=True,reconnect_timeout=20)
ui.run(native=True,window_size=(1600, 900),favicon='res/icon.png',reconnect_timeout=20)
# ui.run(native=True,window_size=(1600, 900),favicon='res/icon.png',dark=True,reconnect_timeout=20)