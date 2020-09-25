
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import yfinance as yf
import plotly.graph_objects as go

def stock_extractor_day(start_date): 

    """
    take the adjusted close price for all stocks.

    from defined start date to today 

    return both the dataframe and the correaltion table 
    """
    tickers = ['FB','GOOGL','MSFT','AMZN','AAPL'] 

    adj_close = yf.download(tickers,start_date)['Adj Close']
    
    fig = go.Figure()
    for col in adj_close.columns: 
        fig.add_trace(go.Scatter(y=adj_close[col], x = list(adj_close.index), mode="lines", name = col))
    fig.show()
    
    return adj_close 
