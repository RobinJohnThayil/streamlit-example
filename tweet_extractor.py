
import twint
import pandas as pd
import numpy as np


def extract_tweets(keyword):
  config = twint.Config()
  config.Search = keyword
  config.Lang = "en"
  config.Limit = 10000
  #config.Since = '2021–04–29'
  #config.Until = '2020–04–29'
  #config.Store_csv = True
  #config.Store_json = True
  config.Output = "pollen.csv"
  #running search
  config.Store_csv = False
  config.Pandas = True


  twint.run.Search(config)


  #c = twint.Config()

  #c.Search = ['#huggies']
  #c.Limit = 10000
  #c.Since = '2022-01-01' 
  #c.Until = '2020-01-01'
  #c.Store_csv = False
  #c.Since = '2021-01-01' 
  #c.Until = '2021-12-31'
  #c.Output = "kc3.csv"

  #twint.run.Search(c)
  df = twint.storage.panda.Tweets_df
  return df
