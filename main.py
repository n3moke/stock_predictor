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
ui.label("Generative AI - Aktienprognose").classes('text-xl')
with ui.tabs().classes('w-full') as tabs:
    one = ui.tab('Auswahl')
    two = ui.tab('Daten')
    three = ui.tab('Prompt')
    four = ui.tab('Prognose')
with ui.tab_panels(tabs, value=one).classes('w-full'):
    with ui.tab_panel(one):
        ui.label('Zeitraum auswählen')
        ui.toggle({1:'1 Monat', 3:'3 Monate',6:' 6 Monate',12: '12 Monate',60: '5 Jahre'},on_change=settings.update_dates).bind_value(settings,'timeframe')
        ui.label('Enddatum wählen')
        with ui.input('Date').bind_value(settings, 'end_date_str') as date:
            with ui.menu().props('no-parent-event') as menu:
                with ui.date(value=settings.end_date_str, on_change=settings.update_dates).bind_value(settings,'end_date_str'):
                    with ui.row().classes('justify-end'):
                        ui.button('Close', on_click=menu.close).props('flat')
            with date.add_slot('append'):
                ui.icon('edit_calendar').on('click', menu.open).classes('cursor-pointer')
        ui.label('Aktienauswahl')
        table = ui.table(
            columns=[
            {'name': 'name', 'label': 'Firmenname', 'field': 'name', 'sortable': True, 'align': 'left'},
            {'name': 'Aktie', 'label': 'Aktienkürzel', 'field': 'aktie', 'sortable': True}],
            rows=dataprovider.aktien_list,
            row_key='name',
            on_select=lambda e: settings.update_aktien(e.selection),
            pagination=10
        )
        table.set_selection('single')
        ui.button('GOTO Daten', on_click=lambda: tabs.set_value(two))
    with ui.tab_panel(two):
        stock_table_container = ui.column()
        with stock_table_container as stock_parent_container:
            aktiendaten_table :ui.table = dataprovider.get_pandas_single_stock_table(settings.aktien, settings.end_date, settings.start_date)

        ui.button('Refresh Stock Data', on_click=lambda _: set_and_update_table())
        goToTabThree =  ui.button('Goto Prompt', on_click=lambda: tabs.set_value(three))
        # ui.switch(value=True, on_change=lambda e: goToTabThree.props(remove='disabled') if e.value else goToTabThree.props('disabled'))
        
                    
        def set_and_update_table():
            if not settings.aktien:
                ui.notify('Please select a stock first to retreive historical data!')
            with stock_table_container as stock_parent_container:
                if not settings.aktien:
                    logger.warning("No stock was selected, while trying to load historical data")
                else:
                    logger.info('deleting children of stock_parent_container and add it again to update it...')
                    stock_parent_container.clear()                
                    df = dataprovider.get_pandas_single_stock(settings.aktien, settings.end_date, settings.start_date)
                    ui.table.from_pandas(df,row_key='Date',pagination={'rowsPerPage': 10},title=settings.aktien[0]['name'])
        
    with ui.tab_panel(three):
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
            ui.button('Do combined forecast', on_click=lambda:  do_combined_forecast()).tooltip('A arima is done with the given timeframe selected in the first tab. The accuracy befenits normally from a longer timeframe.')       
            ui.spinner('dots', size='lg', color='red').bind_visibility(llm, 'isArimaRunning')
     
        def do_sentiment_analysis():
            if not settings.aktien:
                logger.warning('No stock was selected while trying to start a sentiment analysis')
                ui.notify('Please select a in tab 1 stock first!')
            else:
                background_tasks.create(llm.get_sentiment_respone(llm.LLM_TYPE.GEMMA3,settings.aktien,settings.start_date, settings.end_date, settings.newsType))

        def do_arima_forecast():
            if not settings.aktien:
                ui.notify('Please select a in tab 1 stock first!')
            else:
                background_tasks.create(llm.get_arima_response(llm.LLM_TYPE.GEMMA3,settings.aktien,settings.start_date, settings.end_date))

        def do_combined_forecast():
            if not settings.aktien:
                ui.notify('Please select a in tab 1 stock first!')
            else:
                background_tasks.create(llm.get_combined_response(llm.LLM_TYPE.GEMMA3,settings.aktien,settings.start_date, settings.end_date,settings.newsType))
            
    with ui.tab_panel(four):
        ui.label('Prognose')
        
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
                ui.table.from_pandas(llm.sentiment_response)

        ui.button('Load Result', on_click=lambda _: display_predictions())

        def display_predictions():
            set_sentiment_table()
            set_arima_table()
            set_combined_table()
        
        def set_sentiment_table():
            with sent_table_container as sent_parent_container:
                sent_parent_container.clear()
                logger.info(f"Sentiment response: \n {llm.sentiment_response}")
                
                sent_table_title = f"Sentiment Analysis of press releases about {settings.aktien[0]['name']}"
                ui.table.from_pandas(llm.sentiment_response,title=sent_table_title).classes('w-full')


        def set_arima_table():
            with arima_table_container as arima_parent_container:
                arima_parent_container.clear()
                arima_table_title = f"Arima stock forecast for {settings.aktien[0]['name']}"
                ui.table.from_pandas(llm.arima_response,title=arima_table_title).classes('w-full')
        
        def set_combined_table():
             with combined_table_container as combined_parent_container:
                combined_parent_container.clear()
                logger.info(f"Combined response: \n {llm.combined_response}")
                
                sent_table_title = f"Combined Prediction (ARIMA and Sentiment Analysis) of {settings.aktien[0]['name']}"
                ui.table.from_pandas(llm.combined_response,title=sent_table_title).classes('w-full')
                ui.label("Reasoning").bind_text_from(llm, 'combined_reasoning')

                ui.highchart(
                    {
                        'title': False,
                        'plotOptions': {
                            'series': {
                                'stickyTracking': False,
                                'dragDrop': {'draggableY': True, 'dragPrecisionY': 1},
                            },
                        },
                        'series': [
                            {'name': 'A', 'data': [[20, 10], [30, 20], [40, 30]]},
                            {'name': 'B', 'data': [[50, 40], [60, 50], [70, 60]]},
                        ],
                    },
                    extras=['draggable-points'],
                    on_point_click=lambda e: ui.notify(f'Click: {e}'),
                    on_point_drag_start=lambda e: ui.notify(f'Drag start: {e}'),
                    on_point_drop=lambda e: ui.notify(f'Drop: {e}')
                ).classes('w-full h-64')

#ui.run()
ui.run(native=True,window_size=(1600, 900))