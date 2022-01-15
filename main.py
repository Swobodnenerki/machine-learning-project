import sys
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.stats import ttest_ind
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}


def sprawdzenie(ex_name, X, y, train, test):
    if(ex_name == 'PCA'):
        pca = PCA(n_components=1)
        X_train = pca.fit_transform(X[train])
        X_test = pca.transform(X[test])
        return X_train, X_test
    if(ex_name == 'LDA'):
        lda = LDA(n_components=1)
        X_train = lda.fit_transform(X[train], y[train])
        X_test = lda.transform(X[test])
        return X_train, X_test


exs = {
    'PCA': PCA(),
    'LDA': LDA()
}

datasets = ['balance', 'australian']

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats, len(exs)))


for idx, dataset in enumerate(datasets):
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    number_of_features = len(X)
    y = dataset[:, -1].astype(int)
    number_of_features = len(X[0])
    for features_index in range(0, number_of_features):
        features_count = features_index + 1
        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                for ex_id, ex_name in enumerate(exs):
                    ex = clone(exs[ex_name])
                    y_train, y_test = y[train], y[test]
                    X_train, X_test = X[train], X[test]
                    X_train, X_test = sprawdzenie(ex_name, X, y, train, test)
                    clf = clone(clfs[clf_name])
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    scores[clf_id, idx, fold_id, ex_id] = accuracy_score(
                        y_test, y_pred)


np.save('results', scores)
scores = np.load('results.npy')
print("\nScores:\n", scores)
print("\nScores:\n", scores.shape)
# mean_scores = np.mean(scores, axis=2).T
# print("\nMean scores:\n", mean_scores)
