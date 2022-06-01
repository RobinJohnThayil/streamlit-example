import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd

def theTweet(tweet_url):
    api = "https://publish.twitter.com/oembed?url={}".format(tweet_url)
    response = requests.get(api)
    res = response.json()["html"]
    return res

input = st.text_input("Enter your tweet url")
if input:
    res = theTweet(input)
    st.write(res)
    components.html(res,height= 700)
    
d = {
    "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
    "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}
 
e = pd.DataFrame(d)
st.table(data=e)

if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
