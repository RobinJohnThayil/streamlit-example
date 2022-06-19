import io
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from textblob import TextBlob
import gensim
import re
from gensim import corpora
import spacy
import nltk
from nltk.corpus import stopwords
#num_gpus_available = len(tf.config.experimental.list_physical_devices('GPU'))
#!pip install transformers
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import streamlit.components.v1 as components
import requests
from tensorflow.keras.models import load_model
from datetime import datetime, timezone
from pytz import timezone


#@st.cache(allow_output_mutation=True)

    
st.set_page_config(page_title='Social Listening', layout="wide")
    


from PIL import Image
image = Image.open('KC.png')

st.image(image, width = 100)
st.markdown("<h1>Social Listening</h1>", unsafe_allow_html=True)


df_read_kleenex = pd.read_csv("kleenex_extract_2022-06-19.csv") #path folder of the data file
#df_read_kleenex = pd.read_csv("Tweets_extracted.csv") #path folder of the data file
df_read_pampers = pd.read_csv("pampershandle.csv")
df_read_whisper = pd.read_csv("whisperindia.csv")
df=df_read_kleenex.copy()


select_brand = st.selectbox("Select Brands", ["kleenex","Pampers","Whisper"])





#Tweets List complete dataraframe

df_url=df[df['link'].isnull()==False]
url=df_url['link']
df_url_pampers=df_read_pampers[df_read_pampers['link'].isnull()==False]
url_pampers=df_url_pampers['link']

df_url_whisper=df_read_whisper[df_read_whisper['link'].isnull()==False]
url_whisper=df_url_whisper['link']
tweet_list=url.tail(6).tolist()
tweet_list_pampers=url_pampers.head(8).tolist()
tweet_list_whisper=url_whisper.head(8).tolist()
#tweet list cluster 1



def theTweet(tweet_url):
    api = "https://publish.twitter.com/oembed?url={}".format(tweet_url)
    response = requests.get(api)
    res = response.json()["html"]
    return res
    
    
    
    
if select_brand=="kleenex":

    col1, col3, col2 = st.columns([5,1, 5])
    
    col1.subheader("\nInsights Table")
                


    #st.write(df_read_kleenex['tweet'].head(5))
    #col1.write('')


    df_insights_kleenex = pd.read_csv("kleenex_summary_2022-06-19.csv")

    #df_insights_kleenex.drop("Unnamed: 0", axis=1, inplace=True)
    #st.plotly_chart
    #st.dataframe( width=2000, height=768, data = df_insights_kleenex.style.highlight_min(color='pink',axis=0).highlight_max(color='lightgreen', axis=0).set_properties(subset=['Summary'], **{'max-width': '500px'}))
    col1.table(data = df_insights_kleenex.style.highlight_min(subset=['Tweet growth over 1 day (%)', 'Tweet growth over 1 week (%)'], color='pink',axis=0).highlight_max(subset=['Tweet growth over 1 day (%)', 'Tweet growth over 1 week (%)'], color='lightgreen', axis=0).set_properties(subset=['Summary'], **{'max-width': '1000px'}))

    select_topic_kleenex = col2.selectbox("Select Topic", options=df_insights_kleenex["Topic"].unique())
              #default=df1["Topic"].unique()              
             
    col2.subheader("A narrow column with the data")       
    #df_selection=df_insights_kleenex.query("Topic==@select_topic_kleenex")
    col2.write("Insights")
    #st.dataframe(df_selection)  

    if select_topic_kleenex==1:
            
        tweet_topic_1=col2.button('Show me select tweets for Topic 1')
        if tweet_topic_1:
            for each in tweet_list:
                res = theTweet(each)
                col2.write(res,unsafe_allow_html=True)
                components.html(res,height= 700)  

    elif select_topic_kleenex==2:
            
        tweet_topic_2=col2.button('Show me select tweets for Topic 2')
        if tweet_topic_2:
            for each in tweet_list:
                res = theTweet(each)
                st.write(res,unsafe_allow_html=True)
                components.html(res,height= 700) 
    elif select_topic_kleenex==3:
            
        tweet_topic_3=col2.button('Show me select tweets for Topic 3')
        if tweet_topic_3:
            for each in tweet_list:
                res = theTweet(each)
                col2.write(res,unsafe_allow_html=True)
                components.html(res,height= 700)



    #df['created_at'] = df['created_at'].str.replace('India Standard Time', '')
    df.date = pd.to_datetime(df.date)

    line_chart_data=df.copy()
    line_chart_data['created_at']=pd.to_datetime(line_chart_data['created_at'])


    st.write('\nTweets Frequency Visualization ')


    graph=st.button('Tweets Frequency Visualization')
    if graph:
            tweet_df_5min = df.groupby(pd.Grouper(key='date', freq='1440Min', convention='start')).size()
            fig=px.line(tweet_df_5min)
  

            st.plotly_chart(fig)  
                
        


elif select_brand=="Pampers":

    st.write(df_read_pampers['tweet'].head(5))
    st.write('\nInsights Table')
     
    df_insights_pampers=pd.read_csv("topics1.csv")
    df_insights_pampers.drop("Unnamed: 0", axis=1, inplace=True) 
    
    st.write(df_insights_pampers)

    select_topic_pampers = st.selectbox("Select Topic", options=df_insights_pampers["Topic"].unique())
    
    df_selection_pampers=df_insights_pampers.query("Topic==@select_topic_pampers")
    st.write("Insights")
    st.dataframe(df_selection_pampers) 


    if select_topic_pampers==1:
            
        tweet_topic_1_pampers=st.button('Show me select tweets for Topic 1')
        if tweet_topic_1_pampers:
            for each in tweet_list_pampers:
                res = theTweet(each)
                st.write(res,unsafe_allow_html=True)
                components.html(res,height= 700)  

    elif select_topic_pampers==2:
            
        tweet_topic_2_pampers=st.button('Show me select tweets for Topic 2')
        if tweet_topic_2_pampers:
            for each in tweet_list_pampers:
                res = theTweet(each)
                st.write(res,unsafe_allow_html=True)
                components.html(res,height= 700) 
    elif select_topic_pampers==3:
            
        tweet_topic_3_pampers=st.button('Show me select tweets for Topic 3')
        if tweet_topic_3_pampers:
            for each in tweet_list_pampers:
                res = theTweet(each)
                st.write(res,unsafe_allow_html=True)
                components.html(res,height= 700)    
       
    df_read_pampers['created_at'] = df_read_pampers['created_at'].str.replace('India Standard Time', '')
    df_read_pampers.created_at = pd.to_datetime(df_read_pampers.created_at)

    line_chart_data_pampers=df_read_pampers.copy()
    line_chart_data_pampers['created_at']=pd.to_datetime(line_chart_data_pampers['created_at'])


    st.write('\nTweets Frequency Visualization ')


    graph_pampers=st.button('Tweets Frequency Visualization')
    if graph_pampers:
            tweet_df_5min_pampers = df_read_pampers.groupby(pd.Grouper(key='created_at', freq='1440Min', convention='start')).size()
            fig2=px.line(tweet_df_5min_pampers)
  

            st.plotly_chart(fig2)  
            
            
            
elif select_brand=="Whisper":

    st.write(df_read_whisper['tweet'].head(5))
    st.write('\nInsights Table')
     
    df_insights_whisper=pd.read_csv("topics_whisper.csv")
    df_insights_whisper.drop("Unnamed: 0", axis=1, inplace=True) 
    
    st.write(df_insights_whisper)

    select_topic_whisper = st.selectbox("Select Topic", options=df_insights_whisper["Topic"].unique())
    
    df_selection_whisper=df_insights_whisper.query("Topic==@select_topic_whisper")
    st.write("Insights")
    st.dataframe(df_selection_whisper) 


    if select_topic_whisper==1:
            
        tweet_topic_1_whisper=st.button('Show me select tweets for Topic 1')
        if tweet_topic_1_whisper:
            for each in tweet_list_whisper:
                res = theTweet(each)
                st.write(res,unsafe_allow_html=True)
                components.html(res,height= 700)  

    elif select_topic_whisper==2:
            
        tweet_topic_2_whisper=st.button('Show me select tweets for Topic 2')
        if tweet_topic_2_whisper:
            for each in tweet_list_whisper:
                res = theTweet(each)
                st.write(res,unsafe_allow_html=True)
                components.html(res,height= 700) 
    elif select_topic_whisper==3:
            
        tweet_topic_3_whisper=st.button('Show me select tweets for Topic 3')
        if tweet_topic_3_whisper:
            for each in tweet_list_whisper:
                res = theTweet(each)
                st.write(res,unsafe_allow_html=True)
                components.html(res,height= 700)    
       
    df_read_whisper['created_at'] = df_read_whisper['created_at'].str.replace('India Standard Time', '')
    df_read_whisper.created_at = pd.to_datetime(df_read_whisper.created_at)

    line_chart_data_whisper=df_read_whisper.copy()
    line_chart_data_whisper['created_at']=pd.to_datetime(line_chart_data_whisper['created_at'])


    st.write('\nTweets Frequency Visualization ')


    graph_whisper=st.button('Tweets Frequency Visualization')
    if graph_whisper:
            tweet_df_5min_whisper = df_read_whisper.groupby(pd.Grouper(key='created_at', freq='1440Min', convention='start')).size()
            fig3=px.line(tweet_df_5min_whisper)
  

            st.plotly_chart(fig3)  
                   


 
    
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1016A2;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: white;
    color:#000000;
    }
</style>""", unsafe_allow_html=True)
