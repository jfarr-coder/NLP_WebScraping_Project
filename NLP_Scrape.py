import requests
from htmldate import find_date
from bs4 import BeautifulSoup
import pandas as pd

if __name__=="__main__":
    data=pd.read_csv('NLP_Project_Raw_Data_Incomplete.csv')
    data['C/I'] = 'I'
    #date = find_date('https://globalnews.ca/news/8453515/omicron-covid-variant-explained/')
    #print(date)
    #r.encode('utf-8')
    #print(r)
    dates=[]
    #soup = BeautifulSoup(r.content, features='html.parser')
    for l in data['Full Text Link']:
        date = find_date(l)
        dates.append(date)
        r=requests.get(l)
        soup = BeautifulSoup(r.content, features='html.parser')

    data['Date'] = dates
    #print(data['Date'])
    
    
    #print(r.text)