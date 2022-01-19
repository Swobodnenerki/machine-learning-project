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
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

clfs = {
    'GNB-PCA': (GaussianNB(), PCA()),
    'GNB-LDA': (GaussianNB(), LDA()),
    'GNB-KPCA': (GaussianNB(), KernelPCA()),
    'GNB-InPCA': (GaussianNB(), IncrementalPCA()),

    'SVM-PCA': (SVC(), PCA()),
    'SVM-LDA': (SVC(), LDA()),
    'SVM-KPCA': (SVC(), KernelPCA()),
    'SVM-InPCA': (SVC(), IncrementalPCA()),

    'kNN-PCA': (KNeighborsClassifier(), PCA()),
    'kNN-LDA': (KNeighborsClassifier(), LDA()),
    'kNN-KPCA': (KNeighborsClassifier(), KernelPCA()),
    'kNN-InPCA': (KNeighborsClassifier(), IncrementalPCA()),

    'CART-PCA': (DecisionTreeClassifier(random_state=1234), PCA()),
    'CART-LDA': (DecisionTreeClassifier(random_state=1234), LDA()),
    'CART-KPCA': (DecisionTreeClassifier(random_state=1234), KernelPCA()),
    'CART-InPCA': (DecisionTreeClassifier(random_state=1234), IncrementalPCA()),
}


def sprawdzenie(ex_name, X, y, train, test):
    if(ex_name == 'PCA'):
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X[train])
        X_test = pca.transform(X[test])
        return X_train, X_test
    if(ex_name == 'LDA'):
        lda = LDA(n_components=1)
        X_train = lda.fit_transform(X[train], y[train])
        X_test = lda.transform(X[test])
        return X_train, X_test
    if(ex_name == 'KPCA'):
        kpca = KernelPCA(n_components=1, kernel='linear')
        X_train = kpca.fit_transform(X[train], y[train])
        X_test = kpca.transform(X[test])
        return X_train, X_test
    if(ex_name == 'InPCA'):
        inpca = IncrementalPCA(n_components=1, batch_size=10)
        X_train = inpca.fit_transform(X[train])
        X_test = inpca.transform(X[test])
        return X_train, X_test


exs = {
    'PCA': PCA(),
    'LDA': LDA(),
    'KPCA': KernelPCA(),
    'InPCA': IncrementalPCA(),
}

# datasets = ['balance']
datasets = ['balance', 'australian', 'breastcan', 'breastcancoimbra', 'diabetes', 'ecoli4', 'german', 'glass2', 'hayes', 'heart', 'iris', 'liver',
            'monkone', 'monkthree', 'page-blocks-1-3_vs_4', 'soybean', 'wine',
            'wisconsin', 'yeast3', 'yeast-2_vs_8']
n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

with tqdm(total=20) as pbar:
    for idx, dataset in enumerate(datasets):
        dataset = np.genfromtxt("DataSets/%s.csv" % (dataset), delimiter=",")
        X = dataset[:, :-1]
        number_of_features = len(X)
        y = dataset[:, -1].astype(int)
        number_of_features = len(X[0])
        for features_index in range(0, number_of_features):
            features_count = features_index + 1
            for fold_id, (train, test) in enumerate(rskf.split(X, y)):
                for clf_id, clf_name in enumerate(clfs):
                    ex = clone(clfs[clf_name][1])
                    ex_name = clf_name.split('-')[1]
                    y_train, y_test = y[train], y[test]
                    X_train, X_test = X[train], X[test]
                    X_train, X_test = sprawdzenie(
                        ex_name, X, y, train, test)
                    clf = clone(clfs[clf_name][0])
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    scores[clf_id, idx, fold_id] = accuracy_score(
                        y_test, y_pred)
        pbar.update(1)


np.save('results', scores)
scores = np.load('results.npy')
print("\nScores:\n", scores)
print("\nScores:\n", scores.shape)
