
import pandas as pd 
import numpy  as np 
import matplotlib.pyplot as plt 
import plotly.express as px 
import plotly.graph_objects as go

def wiki_extractor(): 
    '''
    returns a dataframe with datetime from 2015, inluding 5 companies'
    daily wikipedia's pageview 

    '''
    place_holder = '/Users/ball4308/Desktop/Facebook stock price ofrecasting/Wiki_data_source'
    tag_list = ['/APPLE_wiki.csv',
                '/AMAZON_wiki.csv',
                '/Facebook_wiki.csv', 
                '/Google_wiki.csv',
                '/Microsoft_wiki.csv']
    df = pd.DataFrame()
   
    for i in range(len(tag_list)) : 

        temp = pd.read_csv(place_holder+ tag_list[i])

        if i == 0: 

            df['Date'] = temp['Date']

            df = df.merge(temp, on = 'Date', how = 'inner')
            df = df.rename(columns = {df.columns[-1]:df.columns[-1] + '.wiki'})
        
        else: 

            df = df.merge(temp, on = 'Date', how = 'inner')
            df = df.rename(columns = {df.columns[-1]:df.columns[-1] + '.wiki'})

    df['Date'] = pd.to_datetime(df['Date'])
    
    fig = go.Figure()

    for col in df.columns[1:]: 
        fig.add_trace(go.Scatter(y=df[col], x = df['Date'],mode="lines",name = col))
    fig.show()

    return df
