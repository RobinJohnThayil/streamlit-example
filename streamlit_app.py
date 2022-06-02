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

def display_tweets():
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
    "Topic": pd.Series([1]),
    "High Fequency Words": pd.Series(["new, know, time"]),
    "Summary": pd.Series([1]),
    "High Fequency Words": pd.Series(["new, know, time"]),
    "Tweet growth over 1 Week": pd.Series(["9.09%"]),
    "Tweet growth rate over 1 Month": pd.Series(["-8.33%"]),
}
 
e = pd.DataFrame(d)
st.table(data=e)
        
if st.button('Show me select tweets for Topic 1'):
    display_tweets()
else:
    st.write('Press ths button to see select tweets for Topic 1')


# class Tweet(object):
#     def __init__(self, s, embed_str=False):
#         if not embed_str:
#             # Use Twitter's oEmbed API
#             # https://dev.twitter.com/web/embedded-tweets
#             api = "https://publish.twitter.com/oembed?url={}".format(s)
#             response = requests.get(api)
#             self.text = response.json()["html"]
#         else:
#             self.text = s

#     def _repr_html_(self):
#         return self.text

#     def component(self):
#         return components.html(self.text, height=1200)


# t = Tweet("https://twitter.com/OReillyMedia/status/901048172738482176").component()

# import streamlit as st
# import streamlit.components.v1 as components

# # bootstrap 4 collapse example
# components.html(
#     """
# <blockquote class="twitter-tweet"><p lang="en" dir="ltr">And we will try to be the first one to get it booked(Hopefully top model with AT Diesel equipped with ventilated seat&#39;s)</p>&mdash; Capt.Ajai K Singh (@SingAjai) <a href="https://twitter.com/SingAjai/status/1532044961604313088?ref_src=twsrc%5Etfw">June 1, 2022</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
#     """,
#     height=600,
# )


