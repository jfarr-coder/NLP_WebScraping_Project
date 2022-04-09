import requests
from htmldate import find_date
from bs4 import BeautifulSoup
import pandas as pd
#import html2text
import nltk
import sklearn

def getDates(filename):
    data=pd.read_csv(filename)
    dates=[]
    data = data.dropna(subset=['Full Text Link'])
    for l in data['Full Text Link']:
        date = find_date(l)
        dates.append(date)
    data['Date'] = dates
    data.to_csv(filename)

if __name__=="__main__":
    getDates('NLP_Project_Raw_Data_Complete.csv')
    getDates('NLP_Project_Raw_Data_Incomplete.csv')