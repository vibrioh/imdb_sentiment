# import pyprind
import pandas as pd
import os
import time
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score

import nltk
from nltk.util import ngrams

train_path = "./aclImdb/train/" # source data
test_path = "./imdb_te.csv" # test data for grade evaluation.

# own_validation = "aclImdb/test/"   # for my own testing accuracy

stopwords = open('./stopwords.en.txt').read().splitlines()


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
	and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have three 
    columns, "row_number", "text" and label'''
    polarities = {'neg':0, 'pos':1}
    # processing_bar = pyprind.ProgBar(25000)
    df = pd.DataFrame() # create a pandas DataFrame object
    for p in polarities:
        path = os.path.join(inpath, p)  # sub-folders of train_path
        for f in os.listdir(path):   # file names in the folders
            file = os.path.join(path, f)
            with open(file, encoding='utf-8') as infile:
                text = infile.read()
                text = re.sub('<[^>]*>', '', text)
                text = re.sub('[\d\W]+', ' ', text.lower())
                df = df.append([[text, polarities[p]]], ignore_index=True)
            # processing_bar.update()
    df.columns = ['text', 'polarity']
    # df.to_csv(os.path.join(outpath, name), index=False) # write for testing
    return df

def sentiment(vectorizer):
    X_train = vectorizer.fit_transform(x_train_list)

    # X_validation = vectorizer.transform(x_validation_list)

    clf = SGDClassifier(loss='hinge', penalty='l1')
    clf.fit(X_train, Y_train)

    # validation_predict = clf.predict(X_validation)
    # train_predict = clf.predict(X_train)
    # print("score_train={}".format(accuracy_score(Y_train, train_predict)))
    # print("score_validation={}".format(accuracy_score(Y_validation, validation_predict)))

    X_test = vectorizer.transform(df_test_list)
    test_predict = clf.predict(X_test)
    return test_predict


def bi_token(text):     # for bigrams using nltk
    token = nltk.word_tokenize(text)
    bigrams = ngrams(token, 2)
    return bigrams


if __name__ == "__main__":
    start = time.clock()
    df_train = imdb_data_preprocess(train_path)
    # df_validation = imdb_data_preprocess(own_validation, name="imdb_own_validation.csv")

    # df_train = pd.read_csv('imdb_tr.csv')  # read while testing, don't need locally read, use memory
    # df_validation = pd.read_csv("imdb_own_validation.csv")
    df_test = pd.read_csv(test_path, encoding='ISO-8859-1')
    df_test_list = df_test['text'].tolist()

    # df_train, df_validation = train_test_split(df_train, test_size=0.1, random_state=23)

    x_train_list = df_train['text'].tolist()
    Y_train = df_train['polarity'].tolist()

    # x_validation_list = df_validation['text'].tolist()
    # Y_validation = df_validation['polarity'].tolist()

    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, 1), min_df=0.00001, max_df=0.99999)
    results = sentiment(vectorizer)
    with open('unigram.output.txt', 'w') as f:
        for res in results:
            f.write("%d\n" % res)

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigramtfidf.output.txt'''
    vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 1), min_df=0.00001, max_df=0.99999)
    results = sentiment(vectorizer)
    with open('unigramtfidf.output.txt', 'w') as f:
        for res in results:
            f.write("%d\n" % res)
     
    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''
    vectorizer = CountVectorizer(tokenizer=bi_token, min_df=0.00001, max_df=0.99999)
    results = sentiment(vectorizer)
    with open('bigram.output.txt', 'w') as f:
        for res in results:
            f.write("%d\n" % res)

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigramtfidf.output.txt'''
    vectorizer = TfidfVectorizer(tokenizer=bi_token, min_df=0.001, max_df=0.999)
    results = sentiment(vectorizer)
    with open('bigramtfidf.output.txt', 'w') as f:
        for res in results:
            f.write("%d\n" % res)


    print('time=',time.clock()-start,'s')














