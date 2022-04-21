import numpy as np
import pandas as pd
import nltk
import sklearn
import requests
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
import stanza

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
#https://www.nltk.org/api/nltk.classify.scikitlearn.html

def tag_sentence(sentence):
    tags = [] 
    terms=[]
    tokenize = sent_tokenize(sentence)
    for i in tokenize:
        # Word tokenizers is used to find the words
        # and punctuation in a string
        wordsList = nltk.word_tokenize(i)

        tagged = nltk.pos_tag(wordsList)
        for t in tagged:
            tags.append(t[1])
            terms.append(t[0])
    return tags, terms

def getDictionary():
    #requests.get("https://github.com/first20hours/google-10000-english/blob/master/20k.txt")
    #requests.get("https://raw.githubusercontent.com/dwyl/english-words/master/words.txt")
    #file1.text
    # open("unigrams.txt","r")
    # open("terms.txt","r")
    # file1.readlines()
    #file1=open("./input/unigrams.txt","r")
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

def getPOS_Tags():
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
def getTermsDataFrame(data,pos_tags,unigrams, bigrams):
    stop_words = set(stopwords.words('english'))
    #features = data['Headline']
    columns=[]
    all_terms=[]
    columns.append('Headline')
    original_size=len(data['Headline'])
    for p in pos_tags:
        probs=[]
        columns.append(p)
        for f in range(original_size):
            s = data['Headline'][f]
            tags,terms=tag_sentence(s)
            terms_num=len(terms) 
            tf = tags.count(p)
            ntf = tf/terms_num
            probs.append(ntf)
            for t in terms:
                all_terms.append(t)
        data[p]=probs

    for u in unigrams:
        probs=[]
        columns.append(u)
        for f in range(original_size):
            s = data['Headline'][f]
            tags,terms=tag_sentence(s)
            terms_num=len(terms) 
            tf = terms.count(u)
            ntf = tf/terms_num
            probs.append(ntf)
        data[u]=probs
    for b in bigrams:
        probs=[]
        columns.append(b)
        for f in range(original_size):
            s = data['Headline'][f]
            tags,terms=tag_sentence(s)
            nltk_tokens = nltk.word_tokenize(s)  	
            terms = list(nltk.bigrams(nltk_tokens))
            terms_num=len(terms) 
            tf = terms.count(b)
            ntf = tf/terms_num
            probs.append(ntf)
        data[b]=probs
    #all_terms = list(set(all_terms))
    #for a in all_terms:
    #    print(a)
    columns.append('C/I')
    #terms = list(set(terms))
    features=data[columns]
    #tags_df = pd.DataFrame(features, index =features,columns =terms)
    return features

if __name__=="__main__":
    complete_training,complete_testing = getData('NLP_Project_Raw_Data_Complete.csv')
    incomplete_training,incomplete_testing = getData('NLP_Project_Raw_Data_Incomplete.csv')

    complete = pd.read_csv('NLP_Project_Raw_Data_Complete.csv').dropna()
    Incomplete = pd.read_csv('NLP_Project_Raw_Data_Incomplete.csv').dropna()

    all_data = pd.concat([complete, Incomplete], axis=0).drop('Answer Portions',axis=1).reset_index()
    #print(all_data)
    label_names = ['Complete','Incomplete']
    labels = all_data['C/I']
    feature_names = ['Headline'] # POS Tag
    #https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
    #https://stackabuse.com/python-for-nlp-parts-of-speech-tagging-and-named-entity-recognition/
    #https://www.nltk.org/book/ch05.html

    #print(getDictionary())
    #terms_df = getTermsDataFrame(all_data,getPOS_Tags(),getDictionary(),getBigrams(all_data))
    #terms_df.to_csv("results.csv")
    nlp = stanza.Pipeline('en')
    doc = nlp("Barack Obama was born in Hawaii.  He was elected President in 2008.")
    print(doc.entities)