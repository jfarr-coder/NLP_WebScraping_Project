import numpy as np
import pandas as pd
import nltk
import sklearn
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


#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
#https://www.nltk.org/api/nltk.classify.scikitlearn.html

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
    feature_names = ['Headline'] # POS Tag
    #https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
    #https://stackabuse.com/python-for-nlp-parts-of-speech-tagging-and-named-entity-recognition/
    #https://www.nltk.org/book/ch05.html
    stop_words = set(stopwords.words('english'))
    features = all_data['Headline']
    print(all_data['Headline'])
    #for f in features:
    #    print(f)
    #    tokenize = sent_tokenize(f)
    #    print(tokenize)

    train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)
    

    #print(train)
    #print(train_labels)
    #print(test)
    #print(test_labels)

    TfidfVectorizer= TfidfVectorizer()
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train['Headline'])
    #print(X_train_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) # DO NOT USE (SVM, Decistion Trees)
    #SVM
    #https://scikit-learn.org/stable/modules/svm.html
    #https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
    #https://towardsdatascience.com/support-vector-machines-explained-with-python-examples-cb65e8172c85
    #Decision Trees
    #https://scikit-learn.org/stable/modules/tree.html
    #https://www.w3schools.com/python/python_ml_decision_tree.asp
    #https://www.datacamp.com/community/tutorials/decision-tree-classification-python
    #https://www.geeksforgeeks.org/decision-tree-implementation-python/
    #SVM Decision Trees
    #https://towardsdatascience.com/a-complete-view-of-decision-trees-and-svm-in-machine-learning-f9f3d19a337b
    #https://towardsdatascience.com/ensemble-learning-with-support-vector-machines-and-decision-trees-88f8a1b5f84b
    #https://www.numpyninja.com/post/a-simple-introduction-to-decision-tree-and-support-vector-machines-svm
    #https://www.codementor.io/blog/text-classification-6mmol0q8oj
    #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    #print(X_train_tfidf)
    #clf = MultinomialNB().fit(X_train_tfidf, X_train_counts)
    #print(clf)