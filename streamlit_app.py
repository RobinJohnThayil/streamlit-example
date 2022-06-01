import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd

def theTweet(tweet_url):
    api = "https://publish.twitter.com/oembed?url={}".format(tweet_url)
    response = requests.get(api)
    res = response.json()["html"]
    return res

# input = st.text_input("Enter your tweet url")


res = theTweet("https://twitter.com/SingAjai/status/1532044961604313088")
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

    
https://twitter.com/SingAjai/status/1532044961604313088
https://twitter.com/jains_mahindra/status/1532044361906544640
https://twitter.com/SingAjai/status/1532044216649797632
https://twitter.com/NileshT72253655/status/1532039503115735040
https://twitter.com/tskathirvel1/status/1532037847040987137
https://twitter.com/iamasrikal/status/1532034759139807232
https://twitter.com/harijames007/status/1532033644465823745
https://twitter.com/GorakalaBhaskar/status/1532033507869921280
https://twitter.com/hjnikki/status/1532032637392416769
https://twitter.com/HUNTDAILYNEWS1/status/1532019548181905409
