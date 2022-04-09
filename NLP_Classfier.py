import numpy as np
import pandas as pd
#import html2text
import nltk
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
def getLabelsFeatures(dataframe):
    label_names = ['Complete','Incomplete']
    label = dataframe['C/I']
    feature_names = ['Headline','C/I']
    features = dataframe[feature_names]
    return label_names, label, feature_names, features

def getALLData(filename, filename2):
    data=pd.read_csv(filename).dropna()
    data2=pd.read_csv(filename2).dropna()
    all_data = pd.concat([data, data2], axis=0).drop('Answer Portions',axis=1)
    return all_data

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


    complete = pd.read_csv('NLP_Project_Raw_Data_Complete.csv').dropna()
    Incomplete = pd.read_csv('NLP_Project_Raw_Data_Incomplete.csv').dropna()

    all_data = pd.concat([complete, Incomplete], axis=0).drop('Answer Portions',axis=1)
    #print(all_data)
    label_names = ['Complete','Incomplete']
    labels = all_data['C/I']
    feature_names = ['Headline','C/I']
    features = all_data[feature_names]

    train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)
    

    print(train)
    print(train_labels)
    print(test)
    print(test_labels)

    #TfidfVectorizer= TfidfVectorizer()
    #count_vect = CountVectorizer()
    #X_train_counts = count_vect.fit_transform(train)
    #print(X_train_counts)
    #tfidf_transformer = TfidfTransformer()
    #X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #print(X_train_tfidf)