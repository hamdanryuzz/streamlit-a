import streamlit as sl
import snscrape.modules.twitter as sntwitter
import pandas as pd
import time
import emoji
import contractions
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import googletrans
from googletrans import *
import matplotlib.pyplot as plt

query = "(ganjar OR anies) lang:id until:2022-09-01 since:2022-01-01"
tweets = []
limit = 3

sl.title('Scrape Dataset')
bScrape = sl.button('Scraping Tweet')
def scrape():
    if bScrape:
        sl.write("Mulai Scraping")
        sl.write("********")
        for tweet in sntwitter.TwitterSearchScraper(query=query).get_items():
            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.date, tweet.user.username, tweet.content])
        df = pd.DataFrame(tweets, columns=['date','username','content'])
        sl.write("Finished")
        sl.write("********")
        sl.write("Berhasil mengambil tweet sebanyak")
        sl.write(df.count(axis=0))
        
        sl.download_button('Download Dataset',df.to_csv(),mime = 'text/csv')
    else:
        pass 
scrape()
sl.write('*******')

sl.title('Upload Dataset')
uploaded_file = sl.file_uploader(label='Upload your dataset.',
            type=['csv','xlsx'])

global df2
if uploaded_file is not None:
    print(uploaded_file)
    print('hello')
    
    try:
        df2 = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df2 = pd.read_excel(uploaded_file)
sl.write(df2)



def clean(tweet):
    #Replace RT
    t1 = re.sub('RT\s', '', tweet)
    #Replace @
    t2 = re.sub('\B@\w+', '', t1)
    #Replace Emoji
    t3 = emoji.demojize(t2)
    #Replace URL
    t4 = re.sub('(http|https):\/\/\S+','',t3)
    #Replace Hastag
    t5 = re.sub('#+', '', t4)
    #all lower
    t6 = t5.lower()
    #Replace repetition word
    t7 = re.sub(r'(.)\1+', r'\1\1', t6)
    #replace symbol repetition
    t8 = re.sub(r'[\?\.\!]+(?=[\?.\!])','',t7)
    #Alphabaet, del number and symbols
    t9  =re.sub(r'[^a-zA-Z]', ' ', t8)
    #replace contractions
    t10 = contractions.fix(t9)
    return t9

for i, r in df2.iterrows():
    y = clean(r['content'])
    df2.loc[i,'content'] = y

sl.title('Cleansing Data')
sClean = sl.button('Show Clean Data')
if sClean:
    sl.write(df2.head())



translator = googletrans.Translator()

df2['content'] = df2['content'].astype(str) #changing datatype to string
df2['trans_content'] = df2['content'].apply(translator.translate, src='id', dest='en').apply(getattr, args=('text',))

sl.title('Translate Data')
sTrans = sl.button('Show Translate Data')
if sTrans:
    sl.write(df2.head())
    

pos_dict = {'J':wordnet.ADJ,'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('indonesian')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

df2['tokenize'] = df2['trans_content'].apply(token_stop_pos)
sl.title('Tokenie')    
sl.write(df2.head())

wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize(pos_data):
    lemma_r = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemmar_r = lemma_r + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_r = lemma_r + " " + lemma
    return lemma_r
df2['Lemma']=df2['tokenize'].apply(lemmatize)

sl.title('Lemmatie')    
sl.write(df2.head())

def getSubjectivy(review):
    return TextBlob(review).sentiment.subjectivity

def getPolarity(review):
    return TextBlob(review).sentiment.polarity

def analys(score):
    if score < 0:
        return 'Negatif'
    elif score == 0:
        return 'Apatis'
    else:
        return 'Positif'

final_data = pd.DataFrame(df2[['date','username','Lemma']])

final_data['Subjectivy'] = final_data['Lemma'].apply(getSubjectivy)
final_data['Polarity'] = final_data['Lemma'].apply(getPolarity)
final_data['TextBlob'] = final_data['Polarity'].apply(analys)

sl.title('Final Data')    
sl.write(final_data.head())

tb_counts = final_data.TextBlob.value_counts()
sl.title('Count Data')    
sl.write(tb_counts)

piee = plt.figure(figsize=(10,7))
pie = plt.pie(tb_counts.values, labels=tb_counts.index, explode=(0,0,0.25), autopct='%1.1f%%', shadow=False)
sl.title('Final Sentimen Analysis')
sl.write(pie)
sl.write(piee)

