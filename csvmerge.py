import pandas as pd
import glob

def merge_csv_With_path_andTimeframe(timeframe: int,stockName: str ,llmName: str,method:str):

    csv_files = glob.glob(f'results/{stockName}/{llmName}/{timeframe}m/*.csv')
    merged = ""
    dfs = []
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(file)
        df.rename(columns={'prediction': f'prediction_{idx+1}'}, inplace=True)
        dfs.append(df)

    from functools import reduce
    merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
    print(merged)
    merged.to_csv(f'results/{stockName}_{llmName}_{timeframe}m_{method}.csv', sep=';', decimal=',', index=False)


if __name__ == '__main__': 
    # stockName = 'NVDA'
    # llm_name = 'llama'
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
    stockName = 'MSFT'
    llm_name = 'llama'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=12,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=60,llmName=llm_name,method=method)
    llm_name = 'gemma'
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=1,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=3,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=12,llmName=llm_name,method=method)
    merge_csv_With_path_andTimeframe(stockName=stockName,timeframe=60,llmName=llm_name,method=method)
