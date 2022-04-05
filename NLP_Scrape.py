import requests
from htmldate import find_date
from bs4 import BeautifulSoup
import pandas as pd
import html2text
#import nltk

if __name__=="__main__":
    data=pd.read_csv('NLP_Project_Raw_Data_Complete.csv')
    #date = find_date('https://globalnews.ca/news/8453515/omicron-covid-variant-explained/')
    #print(date)
    #r.encode('utf-8')
    #print(r)
    dates=[]
    #ftexts=[]
    data = data.dropna(subset=['Full Text Link'])
    #soup = BeautifulSoup(r.content, features='html.parser')
    for l in data['Full Text Link']:
        #ftext=""
        date = find_date(l)
        dates.append(date)
        #r=requests.get(l)
        #h=html2text.HTML2Text()
        #h.ignore_links=True
        #soup = BeautifulSoup(r.content, features='html.parser')
        #ftext= soup.get_text()
        #ftexts.append(ftext.strip().encode("utf-8").decode("utf-8"))

    data['Date'] = dates
    #data['Full Text'] = ftexts
    #data.to_csv('NLP_Project_Raw_Data_Complete.csv')
    
    
    #print(r.text)