import requests
from htmldate import find_date
from bs4 import BeautifulSoup
import pandas as pd
import html2text

if __name__=="__main__":
    data=pd.read_csv('NLP_Project_Raw_Data_Incomplete.csv')
    #date = find_date('https://globalnews.ca/news/8453515/omicron-covid-variant-explained/')
    #print(date)
    #r.encode('utf-8')
    #print(r)
    #dates=[]
    ftexts=[]
    #soup = BeautifulSoup(r.content, features='html.parser')
    for l in data['Full Text Link']:
        ftext=""
        #date = find_date(l)
        #dates.append(date)
        r=requests.get(l)
        h=html2text.HTML2Text()
        h.ignore_links=True
        soup = BeautifulSoup(r.content, features='html.parser')
        ftext= h.handle(r.text)
        #ftexts.append(ftext)

    #data['Date'] = dates
    data['Full Text'] = ftexts
    #data.to_csv('NLP_Project_Raw_Data_Incomplete.csv')
    
    
    #print(r.text)