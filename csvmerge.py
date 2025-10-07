import pandas as pd
import glob

def merge_csv_With_path_andTimeframe(timeframe: int,stockName: str ,llmName: str,method:str):

    csv_files = glob.glob(f'results/{stockName}/{timeframe}M/*.csv')
    merged = ""
    dfs = []
    print(csv_files)
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(file, delimiter=';', decimal=',')
        df.rename(columns={'prediction': f'prediction_{idx+1}'}, inplace=True)
        dfs.append(df)
    # print(dfs)
    from functools import reduce
    merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
    merged.to_csv(f'results/{stockName}_{llmName}_{timeframe}m_{method}.csv', sep=';', decimal=',', index=False)


if __name__ == '__main__': 
    # stockName = 'NVDA'
    llm_name = 'llama'
    method = 'combined'
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=12,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=60,llmName=llm_name,method=method)
    # llm_name = 'gemma'
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=12,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=60,llmName=llm_name,method=method)
    stockName = 'AAPL'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=12,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=60,llmName=llm_name,method=method)
    # llm_name = 'gemma'
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=12,llmName=llm_name,method=method)
    # merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=60,llmName=llm_name,method=method)

    stockName = 'TSLA'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    stockName = 'NVDA'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    stockName = 'NFLX'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    stockName = 'MSFT'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    stockName = 'META'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    stockName = 'GOOGL'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    stockName = 'GOOG'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    stockName = 'AVGO'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    stockName = 'AMZN'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
