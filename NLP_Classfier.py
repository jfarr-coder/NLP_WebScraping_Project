import numpy as np
import pandas as pd
import nltk
import sklearn
import requests
import re
import os
import json
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.parse import stanford
from nltk.parse.corenlp import CoreNLPServer
from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
import stanza
import StanfordDependencies
from pprint import pprint
#from pycorenlp.corenlp import StanfordCoreNLP
from stanfordcorenlp import StanfordCoreNLP

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
#https://www.nltk.org/api/nltk.classify.scikitlearn.html

def term_sentence(sentence):
    terms=[]
    tokenize = sent_tokenize(sentence)
    for i in tokenize:
        # Word tokenizers is used to find the words
        # and punctuation in a string
        wordsList = nltk.word_tokenize(i)

        tagged = nltk.pos_tag(wordsList)
        for t in tagged:
            terms.append(t[0])
    return terms

def tag_sentence(sentence):
    tags = [] 
    terms=[]
    regex=""
    POS_Parser = getTags()
    for t in range(len(POS_Parser)):
        regex+=POS_Parser[t]
        regex+="|"
    nlp = StanfordCoreNLP('http://localhost', port=9002)
    stanford = nlp.parse(sentence)
    tags = re.findall(regex, stanford)
    tags.remove('')
    nlp.close()
    return tags

def getDictionary():
    file1 =open("./input/terms.txt","r")
    unigrams=[]
    for l in file1.readlines():
        unigrams.append(l.strip())
    return unigrams

def getBigrams(data):
    bigrams=[]
    original_size=len(data['Headline'])
    for f in range(original_size):
        s = data['Headline'][f]
        nltk_tokens = nltk.word_tokenize(s)  	
        terms = list(nltk.bigrams(nltk_tokens))
        for t in terms:
            bigrams.append(t)
    return bigrams

def getTags():
    tags=[]
    file1=open("./input/POS_Tags.txt","r")
    for l in file1.readlines():
        tags.append(l.strip())
    return tags

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

#https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
#https://www.geeksforgeeks.org/python-ways-to-remove-duplicates-from-list/

def getTagsDataFrame(data,pos_tags):
    data2=data.copy().drop('Headline',axis=1).drop('C/I',axis=1)
    stop_words = set(stopwords.words('english'))
    columns=[]
    all_terms=[]
    original_size=len(data['Headline'])
    for p in pos_tags:
        probs=[]
        columns.append(p)
        for f in range(original_size):
            s = data['Headline'][f]
            tags=tag_sentence(s)
            terms=term_sentence(s)
            terms_num=len(terms) 
            tf = tags.count(p)
            ntf = tf/terms_num
            probs.append(ntf)
        data2[p]=probs
    tags_df=data2[columns]
    return tags_df

def getTermsDataFrame(data,unigrams):
    data2=data.copy().drop('Headline',axis=1).drop('C/I',axis=1)
    stop_words = set(stopwords.words('english'))
    columns=[]
    all_terms=[]
    original_size=len(data['Headline'])
    for u in unigrams:
        probs=[]
        columns.append(u)
        for f in range(original_size):
            s = data['Headline'][f]
            terms=term_sentence(s)
            terms_num=len(terms) 
            tf = terms.count(u)
            ntf = tf/terms_num
            probs.append(ntf)
        data2[u]=probs
    unigrams_df=data2[columns]
    return unigrams_df

def getBigramsDataFrame(data,bigrams):
    data2=data.copy().drop('Headline',axis=1).drop('C/I',axis=1)
    stop_words = set(stopwords.words('english'))
    columns=[]
    all_terms=[]
    original_size=len(data['Headline'])
    for b in bigrams:
        probs=[]
        columns.append(b)
        for f in range(original_size):
            s = data['Headline'][f]
            terms=term_sentence(s)
            nltk_tokens = nltk.word_tokenize(s)  	
            terms = list(nltk.bigrams(nltk_tokens))
            terms_num=len(terms) 
            tf = terms.count(b)
            ntf = tf/terms_num
            probs.append(ntf)
        data2[b]=probs
    bigrams_df=data2[columns]
    return bigrams_df

if __name__=="__main__":
    complete_training,complete_testing = getData('NLP_Project_Raw_Data_Complete.csv')
    incomplete_training,incomplete_testing = getData('NLP_Project_Raw_Data_Incomplete.csv')

    complete = pd.read_csv('NLP_Project_Raw_Data_Complete.csv').dropna()
    Incomplete = pd.read_csv('NLP_Project_Raw_Data_Incomplete.csv').dropna()

    all_data = pd.concat([complete, Incomplete], axis=0).drop('Answer Portions',axis=1).reset_index()
    #print(all_data)
    label_names = ['Complete','Incomplete']
    labels = all_data['C/I']
    headlines = all_data['Headline'] # POS Tag
    #https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
    #https://stackabuse.com/python-for-nlp-parts-of-speech-tagging-and-named-entity-recognition/
    #https://www.nltk.org/book/ch05.html

    tags_df=getTagsDataFrame(all_data,getTags())
    results=pd.concat([headlines, tags_df], axis=1)

    terms_df = getTermsDataFrame(all_data,getDictionary())
    results=pd.concat([results, terms_df], axis=1)
    
    bigrams_df=getBigramsDataFrame(all_data,getBigrams(all_data))
    result=pd.concat([results, bigrams_df], axis=1)
    
    result=pd.concat([results, labels], axis=1)

    #results.to_csv("results.csv")