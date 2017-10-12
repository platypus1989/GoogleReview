import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from __future__ import division
from __future__ import print_function
import os
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.book import *
from nltk.corpus import stopwords
from time import time
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



os.getcwd()
review_data = pd.read_csv('Documents/GoogleReview/data/review_data_v1.csv')

review_data['review_text'] = review_data['review_text'].astype(str)
review_text = ' '.join(review_data['review_text'])
review_text = review_text.lower()

review_data.review_rating.value_counts()

all_rating = set(review_data.review_rating)
stop = stopwords.words('english')
stop = stop + ['.', 'I', ',', 'n\'t', '(', ')']
stop_set = set(stop)


n = 100

def keywordExtract(text, number):
    all_text = ' '.join(text)
    tokens = nltk.word_tokenize(all_text.decode('utf-8').lower())
    tokens = [token for token in tokens if token not in stop_set]
    fdist = FreqDist(tokens)
    return [i[0] for i  in fdist.most_common()[:number]]

rating_keywords = pd.DataFrame(np.zeros([n, len(all_rating)]))
    
for i, rating in enumerate(all_rating):
    text = review_data.review_text[review_data.review_rating == rating]
    rating_keywords[rating] = keywordExtract(text, n)

feature_matrix = np.zeros([review_data.shape[0], len(all_rating)])

tic = time()
for i, review in enumerate(review_data['review_text']):
    for j in range(len(all_rating)):    
        name_tokens = nltk.word_tokenize(review.decode('utf-8').lower())
        feature_matrix[i,j] = sum(name_token in list(rating_keywords.iloc[:,j]) for name_token in name_tokens)
toc = time() 
toc - tic
           
y = review_data['review_rating'].astype('category')


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "GBM", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GradientBoostingClassifier(),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X = feature_matrix
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)


for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(score, name)

#==============================================================================
# 0.452991452991 Nearest Neighbors
# 0.509615384615 Linear SVM
# 0.530982905983 RBF SVM
# 0.543803418803 Gaussian Process
# 0.517094017094 Decision Tree
# 0.510683760684 Random Forest
# 0.521367521368 GBM
# 0.513888888889 Neural Net
# 0.519230769231 AdaBoost
# 0.478632478632 Naive Bayes
# 0.113247863248 QDA
#==============================================================================


good_words = ['great', 'good', 'nice', 'friendly', 'best', 'delicious', 'really',
              'love', 'excellent', 'well', 'tasty', 'amazing', 'perfect', 'happy',
              'recommend', 'fresh']
neutral_words = ['staff', 'always', 'also', 'even', 'quality', 'everything','family',
                 'clean', 'decent', 'price', 'pretty', 'will','highly','waiter']
bad_words = ['bad', 'waited', 'never','worst']

key_words = good_words + neutral_words + bad_words

X1 = np.zeros([review_data.shape[0],len(key_words)], dtype=bool)

for i, word in enumerate(key_words):
    for j, review in enumerate(review_data['review_text']):
        if re.search(word,review.lower()):
            X1[j,i] = True

X = np.concatenate([feature_matrix, X1], axis=1)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)


for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(score, name)

#==============================================================================
# 0.46688034188 Nearest Neighbors
# 0.568376068376 Linear SVM
# 0.501068376068 RBF SVM
# 0.113247863248 Gaussian Process
# 0.525641025641 Decision Tree
# 0.491452991453 Random Forest
# 0.582264957265 GBM
# 0.575854700855 Neural Net
# 0.550213675214 AdaBoost
# 0.255341880342 Naive Bayes
# 0.223290598291 QDA
#==============================================================================
