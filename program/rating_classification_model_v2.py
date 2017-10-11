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


X = np.zeros([review_data.shape[0],len(key_words)], dtype=bool)

for i, word in enumerate(key_words):
    for j, review in enumerate(review_data['review_text']):
        if re.search(word,review.lower()):
            X[j,i] = True
            
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


X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)


for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(score, name)

#==============================================================================
# 0.450854700855 Nearest Neighbors
# 0.532051282051 Linear SVM
# 0.518162393162 RBF SVM
# 0.554487179487 Gaussian Process
# 0.498931623932 Decision Tree
# 0.487179487179 Random Forest
# 0.561965811966 GBM
# 0.569444444444 Neural Net
# 0.55235042735 AdaBoost
# 0.247863247863 Naive Bayes
# 0.239316239316 QDA
#==============================================================================

