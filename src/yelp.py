#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:41:45 2018

@author: qiannanmiao
"""

from gensim import corpora, models, similarities

from nltk.corpus import stopwords
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer # slightly better than porter stemmer
from nltk.probability import FreqDist

import pandas as pd
import numpy as np
import string

from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import time, datetime

# helper method for readability measures
# https://datawarrior.wordpress.com/2016/03/29/flesch-kincaid-readability-measure/
not_punctuation = lambda w: not (len(w)==1 and (not w.isalpha()))
get_word_count = lambda text: len(list(filter(not_punctuation, word_tokenize(text))))
get_sent_count = lambda text: len(sent_tokenize(text))

prondict = cmudict.dict()
numsyllables_pronlist = lambda l: len(list(filter(lambda s: chr(s.encode('ascii', 'ignore').lower()[-1]).isdigit(), l)))

def numsyllables(word):
    try:
        return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
    except KeyError:
        return [0]

def text_statistics(text):
    word_count = get_word_count(text)
    sent_count = get_sent_count(text)
    syllable_count = sum(map(lambda w: max(numsyllables(w)), word_tokenize(text)))
    return word_count, sent_count, syllable_count

flesch_formula = lambda word_count, sent_count, syllable_count : 206.835 - 1.015*word_count / sent_count - 84.6 * syllable_count / word_count
def flesch(text):
    word_count, sent_count, syllable_count = text_statistics(text)
    if sent_count > 0 and word_count > 0:
        return flesch_formula(word_count, sent_count, syllable_count)
    else:
        return 206.835

fk_formula = lambda word_count, sent_count, syllable_count : 0.39 * word_count / sent_count + 11.8 * syllable_count / word_count - 15.59
def flesch_kincaid(text):
    word_count, sent_count, syllable_count = text_statistics(text)
    if sent_count > 0 and word_count > 0:
        return fk_formula(word_count, sent_count, syllable_count)
    else:
        return -15.59
    

df = pd.read_csv("/Users/qiannanmiao/Downloads/yelp_review.csv")

# for efficiency, only using 4000 samples for this project
data = df.sample(4000)

# some features for reviews:
length = []
word_count = []
flesch_score = []
flesch_kincaid_score = []
text_space = [] # vector space of each review text after removing stop words and stemming

# some other possible features:
# those are some of the most frequent words in positive reviews and are not frequent
# in negative reviews. After verifying, among all reviews containing one of these words,
# around 80% of them are positive reviews
poswords = ["delicious", "love", "friendly", "best", "amazing"]
has_poswords = []


# classification for reviews:
# is a review has more than 3 stars, then it's classified as positive,
# otherwise, it's classified as negative
# is_positive = [1 if x > 3 else 0 for x in data['stars']]
is_positive = []

# Stop Words:
stop_words = set(stopwords.words("english"))

# Stemmer:
stemmer = SnowballStemmer("english")

# FreqDist() for counting most frequent words in positive and negative reviews
pos_fdist = FreqDist()
neg_fdist = FreqDist()

# Prepare input features for the given dataset
for stars, text in zip(data.stars, data.text):
    
    # get text of review, convert to lowercase
    tmp_text = text.lower()
    
    # Length:
    length.append(len(tmp_text))
    
    if sum([1 if x in tmp_text else 0 for x in poswords]) > 0:
        has_poswords.append(1)
    else:
        has_poswords.append(0)
    
    # Word Count:
    # define a regex for tokenenize tmp_text into array of words
    pattern = r'''(?x)           # set flag to allow verbose regexps
    (?:[A-Z']\.)+(?:'\w*)?       # abbreviations, e.g. U.S.A.
    | \w+(?:[-']\w+)*            # words with optional internal hyphens
    | \$?\d+(?:\.\d+)?%?         # currency and percentages, e.g. $12.40, 82%
    '''
    word_count.append(len(regexp_tokenize(tmp_text, pattern)))
    
    # Readability:
    if len(tmp_text.strip()) > 0:
        flesch_score.append(flesch(tmp_text))
        flesch_kincaid_score.append(flesch_kincaid(tmp_text))
    else:
        flesch_score.append(206.835)
        flesch_kincaid_score.append(-15.59)
    
    
    # NLP preprocess for Dictionary construction:
    # remove stop words
    # replace punctuations by space in tmp_text
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    tmp_text = tmp_text.translate(replace_punctuation)
    words = word_tokenize(tmp_text)
    words_filtered = [w for w in words if not w in stop_words]
    
    if stars > 3:
        is_positive.append(1)
        for word in words_filtered:
            pos_fdist[word] += 1
    else:
        is_positive.append(0)
        for word in words_filtered:
            neg_fdist[word] += 1
    
    # stemming
    for i in range(len(words_filtered)):
        words_filtered[i] = stemmer.stem(words_filtered[i])
        
    text_space.append(words_filtered)
    
    
# count how many reviews with positive words are classified as positive reviews   
count = 0
for i, j in zip(has_poswords, is_positive):
    if i == 1 and j == 1:
        count += 1
print("Review with positive words that are classified as postivie review:", count/sum(has_poswords))

data.index = data.review_id
# add the features extracted to data
data['length'] = length
data['word_count'] = word_count
data['flesch_score'] = flesch_score
data['flesch_kincaid_score'] = flesch_kincaid_score
data['has_poswords'] = has_poswords

data['text_space'] = text_space

data['is_positive'] = is_positive

# some other helpful features:
data['timediff'] = 0.0
data['avg_sim'] = 0.0
data['min_sim'] = 0.0
data['max_sim'] = 0.0

# reviews for the same business might have 
for bid in np.unique(data.business_id):
    
    # get all reviews for this business
    tmp_review_data = data.loc[(data.business_id == bid)]
    
    for rid, rdate in zip(tmp_review_data.review_id, tmp_review_data.date):
        rdate = datetime.datetime.strptime(rdate, "%Y-%m-%d").timetuple()
        today = datetime.date.today().timetuple()
        
        data.at[(rid, "timediff")] = time.mktime(today) - time.mktime(rdate)
    
    # construct a dictionary with the reviews for the current business for similarity measures
    dictionary = corpora.Dictionary(tmp_review_data.text_space)
    
    if len(dictionary.token2id) > 0:
        corpus = [dictionary.doc2bow(rev) for rev in tmp_review_data.text_space]
        
        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
        index = similarities.MatrixSimilarity(lsi[corpus])
        
        for rid, rev in zip(tmp_review_data.review_id, tmp_review_data.text_space):
            rev_vec = dictionary.doc2bow(rev)
            sims = index[lsi[rev_vec]]
            
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            sims_new = [x[1] for x in sims[1:]]
            
            if len(sims_new) > 0:
                data.at[(rid, "avg_sim")] = np.mean(sims_new)
                data.at[(rid, "min_sim")] = np.min(sims_new)
                data.at[(rid, "max_sim")] = np.max(sims_new)

data['order'] = 0.0
for bid in np.unique(data.business_id):
    tmp_review_data = data.loc[(data.business_id == bid)]
    times = tmp_review_data.timediff.tolist()
    times.sort()
    for rid in tmp_review_data.index:
        data.at[(rid, "order")] = times.index(data.at[(rid, "timediff")]) + 1

# random split for train-test
train, test = train_test_split(data, test_size=0.2)

# exclude the following columns that are not needed for the model
excludes = ['review_id', 'user_id', 'business_id', 'stars', 'date', 'text', 'text_space', 'timediff', 'is_positive']
X_train = train.drop(excludes, axis=1)
y_train = train.is_positive

X_test = test.drop(excludes, axis=1)
y_test = test.is_positive



# Helper method for plotting validation curve for a given model and given parameters
def plot_validation_curve(estimator, title, X, y, param_name, param_range, logx):

    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name,
                                                 param_range=param_range, cv=5,
                                                 scoring=make_scorer(cohen_kappa_score, weights="quadratic"))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    lw = 2
    if logx:
        plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    else:
        plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    if logx:
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    else:
        plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    plt.legend(loc="best")
    return plt

# Helper method for plotting the learning curve of a given estimator
def plot_learning_curve(estimator, title, X, y):
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5,
                                                            train_sizes=np.linspace(.1, 1.0, 5),
                                                            scoring=make_scorer(cohen_kappa_score, weights="quadratic"))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt


# 5-fold cross-validation to get the best max_depth
param_range = [i for i in range(2, 21) if i % 2 == 0]
title = "Validation Curve with Random Forest"
plot_validation_curve(RandomForestClassifier(), title, X_train, y_train, "max_depth", param_range, False)
plt.show()

# Plot learning curve:
title = "Learning Curves (Random Forest Classifier with max_depth=8)"
plot_learning_curve(RandomForestClassifier(max_depth=10), title, X_train, y_train)
plt.show()

# From validation above, max_depth=8 gives best score
# Random Forest Classifier    
rf_model = RandomForestClassifier(max_depth=8, random_state=0)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

accuracy = 0
for y_pred, y in zip(rf_pred, y_test):
    if y_pred == y:
        accuracy += 1
print("Accuracy: " + str(accuracy/len(y_test)))
# Accuracy = 73%

for feature, importance in zip(X_train.columns, rf_model.feature_importances_):
    print(feature, ": ", importance)


km_model = KMeans(n_clusters=2, random_state=0)
km_model.fit(X_train)

km_pred = km_model.predict(X_test)

accuracy = 0
for y_pred, y in zip(km_pred, y_test):
    if y_pred == y:
        accuracy += 1
print("Accuracy: " + str(accuracy/len(y_test)))
# Accuracy = 33%


