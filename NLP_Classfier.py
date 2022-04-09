import requests
from htmldate import find_date
from bs4 import BeautifulSoup
import pandas as pd
#import html2text
import nltk
import sklearn

def getData(filename):
    data=pd.read_csv(filename)
    data=data.dropna()
    training_data=data[:60]
    testing_data=data[60:]

    return training_data, testing_data
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
    complete_training,complete_testing = getData('NLP_Project_Raw_Data_Complete.csv')
    incomplete_training,incomplete_testing = getData('NLP_Project_Raw_Data_Incomplete.csv')
    label_names = ['Complete','Incomplete']
    label = complete['C/I']
    feature_names = ['Headline','C/I']
    features = complete[feature_names]

    print(complete_training)
    print(complete_testing)

    print(incomplete_training)
    print(incomplete_testing)

    complete = pd.read_csv('NLP_Project_Raw_Data_Complete.csv').dropna()
    Incomplete = pd.read_csv('NLP_Project_Raw_Data_Incomplete.csv').dropna()
    #print(complete)
    #print(Incomplete)
    