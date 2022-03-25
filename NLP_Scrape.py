import requests
from htmldate import find_date
from bs4 import BeautifulSoup
import pandas as pd

if __name__=="__main__":
    data=pd.read_csv('NLP Project Raw Data_Incomplete - Sheet1.csv')
    print(data['C/I'])
    #date = find_date('https://globalnews.ca/news/8453515/omicron-covid-variant-explained/')
    #print(date)
    #r.encode('utf-8')
    #print(r)
    #soup = BeautifulSoup(r.content, features='html.parser')
    #print(r.text)