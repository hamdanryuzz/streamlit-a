import streamlit as sl
import snscrape.modules.twitter as sntwitter
import pandas as pd
import time
query = "(ganjar OR anies) lang:id until:2022-09-01 since:2022-01-01"
tweets = []
limit = 7

try:
    sl.write("Mulai Scraping")
    sl.write("********")
    for tweet in sntwitter.TwitterSearchScraper(query=query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.content])
    df = pd.DataFrame(tweets, columns=['date','username','content'])
    
except Exception as e:
    sl.write(e)

sl.write("Finished")
sl.write("********")
sl.write("Berhasil mengambil tweet sebanyak")
sl.write(df.count(axis=0))
df.to_csv('tweet.csv', index=False)