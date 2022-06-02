import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd

def theTweet(tweet_url):
    api = 'https://publish.twitter.com/oembed?url={}'.format(tweet_url)
    response = requests.get(api)
    res = response.json()["html"]
    return res

# input = st.text_input("Enter your tweet url")

tweet_list = ["https://twitter.com/SingAjai/status/1532044961604313088",
"https://twitter.com/jains_mahindra/status/1532044361906544640",
"https://twitter.com/SingAjai/status/1532044216649797632",
"https://twitter.com/NileshT72253655/status/1532039503115735040",
"https://twitter.com/tskathirvel1/status/1532037847040987137",
"https://twitter.com/iamasrikal/status/1532034759139807232",
"https://twitter.com/harijames007/status/1532033644465823745",
"https://twitter.com/GorakalaBhaskar/status/1532033507869921280",
"https://twitter.com/hjnikki/status/1532032637392416769",
"https://twitter.com/HUNTDAILYNEWS1/status/1532019548181905409"]

for each in tweet_list:
    res = theTweet(each)
    st.write(res, unsafe_allow_html=True)
    # components.html(res,height= 2000)
    
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

    
class Tweet(object):
    def __init__(self, s, embed_str=False):
        if not embed_str:
            # Use Twitter's oEmbed API
            # https://dev.twitter.com/web/embedded-tweets
            api = "https://publish.twitter.com/oembed?url={}".format(s)
            response = requests.get(api)
            self.text = response.json()["html"]
        else:
            self.text = s

    def _repr_html_(self):
        return self.text

    def component(self):
        return components.html(self.text, height=600)


t = Tweet("https://twitter.com/OReillyMedia/status/901048172738482176").component()
