
import io
import pandas as pd
import numpy as np
#from tqdm.auto import tqdm
#import tensorflow as tf
from transformers import BertTokenizer
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
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



#@st.cache(allow_output_mutation=True)

    
st.set_page_config(page_title='Social Listening')
st.markdown("<h1 style='text-align: center; color: white; font-size: 30px; background-color: #1016A2;'>Social Listening</h1>", unsafe_allow_html=True)

df = pd.read_csv("Tweets_extracted (1).csv") #path folder of the data file
st.write(df['tweet'].head(5))

st.write('\nInsights Table')

df1 = pd.read_csv("table_summary.csv") 
df1.drop("Unnamed: 0", axis=1, inplace=True)
st.write(df1)



nltk.download('stopwords')
nltk.download('words')
words = set(nltk.corpus.words.words())
stopwords= stopwords.words("english")

def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", " ", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", " ").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet)
         if w.lower() in words or not w.isalpha())
    return tweet
    

df['tweet_clean'] = df['tweet'].apply(cleaner)

df['tweet_clean'] = df['tweet_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

def lemmatization(tweet_clean, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp=spacy.load("en_core_web_sm", disable=["parser", "ner"])
    tweet_out=[]
    for text in tweet_clean:
        doc=nlp(text)
        new_text=[]
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final=" ".join(new_text)
        tweet_out.append(final)
    return (tweet_out)

lemmatized_tweet=lemmatization(df['tweet_clean'])

def gen_words(tweet_clean):
    final =[]
    for text in tweet_clean:
        new=gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)
data_words=gen_words(lemmatized_tweet)


def remove_emojis(data_words):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data_words)

id2word= corpora.Dictionary(data_words)
corpus=[]
for text in data_words:
    new=id2word.doc2bow(text)
    corpus.append(new)



lda_model= gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=id2word,
                                          num_topics=25,
                                          random_state=100,
                                          update_every=1,
                                          chunksize=100,
                                          passes=10,
                                          iterations=100,
                                          alpha="auto")
#new_model_lda=gensim.models.ldamodel.LdaModel.load("D:/model_lda/test_model.model")





lda_corpus=lda_model[corpus]
from itertools import chain

scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))

threshold = sum(scores)/len(scores)



cluster1 = [j for i,j in zip(lda_corpus,df.index) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,df.index) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,df.index) if i[2][1] > threshold]
df_cluster1=df.iloc[cluster1]
df_cluster2=df.iloc[cluster2]
df_cluster3=df.iloc[cluster3]
#df_cluster1=df.iloc[cluster1]
#print(threshold)


#df_cluster1 = pd.read_csv("D:/cluster1.csv") 
cluster1_tweet=df_cluster1['tweet'].tolist()
#st.write(df_cluster1)

#df_cluster2 = pd.read_csv("D:/cluster2.csv")
cluster2_tweet = df_cluster2['tweet'].tolist()

#df_cluster3 = pd.read_csv("D:/cluster3.csv")

cluster3_tweet = df_cluster3['tweet'].tolist()

#Tweets List complete dataraframe

df_url=df[df['link'].isnull()==False]
url=df_url['link']

tweet_list=url.tail(6).tolist()

#tweet list cluster 1

df_url_cluster1=df_cluster1[df_cluster1['link'].isnull()==False]
url_cluster1=df_url_cluster1['link']

tweet_list_cluster1=url_cluster1.head(6).tolist()

#tweet list cluster 2

df_url_cluster2=df_cluster2[df_cluster2['link'].isnull()==False]
url_cluster2=df_url_cluster2['link']

tweet_list_cluster2=url_cluster2.tail(5).tolist()

#tweet list cluster 3

df_url_cluster3=df_cluster3[df_cluster3['link'].isnull()==False]
url_cluster3=df_url_cluster3['link']

tweet_list_cluster3=url_cluster3.tail(5).tolist()

def theTweet(tweet_url):
    api = "https://publish.twitter.com/oembed?url={}".format(tweet_url)
    response = requests.get(api)
    res = response.json()["html"]
    return res
    
    


#input = tweet_list[1]
 
#if input:
    #res = theTweet(input)
    #st.write(res)
    #components.html(res,height= 700)


result=st.button('Top Tweets in the file')
if result:
        for each in tweet_list:
            res = theTweet(each)
            st.write(res,unsafe_allow_html=True)
            components.html(res,height= 700)
            
option = st.selectbox(
        #"Select Category", ["--ALL--","Delivery","Packaging", "Price","Product","Competition","Repurchase","Recommend"])
              "Select Topic",
              options=df1["Topic"].unique()
              #default=df1["Topic"].unique()              
              )
            
df_selection=df1.query("Topic==@option")
st.write("Insights")
st.dataframe(df_selection)

if option==1:
            
    result1=st.button('Show me select tweets for Topic 1')
    if result1:
        for each in tweet_list_cluster1:
            res = theTweet(each)
            st.write(res,unsafe_allow_html=True)
            components.html(res,height= 700)
        
         

elif option==2:

    result2=st.button('Show me select tweets for Topic 2')
    if result2:
        for each in tweet_list_cluster2:
            res = theTweet(each)
            st.write(res,unsafe_allow_html=True)
            components.html(res,height= 700)
        
        
elif option==3:

    result3=st.button('Show me select tweets for Topic 3')
    if result3:
        for each in tweet_list_cluster3:
            res = theTweet(each)
            st.write(res,unsafe_allow_html=True)
            components.html(res,height= 700)
        
        
        

#image_frequency = Image.open('D:/tweet_frequency.png')

st.write('\nTweets Frequency Visualization ')

#st.image(image_frequency, caption='Tweets Frequency timeline (Huggies)')


            
            
            




    
    
    
    
    
