import sys
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.stats import ttest_ind
from tabulate import tabulate
import matplotlib.pyplot as plt


datasets = ['balance', 'cleveland', 'ecoli', 'glass', 'iris', 'led7digit', 'magic', 'phoneme',
            'pima', 'ring', 'shuttle', 'twonorm', 'wine', 'winequality-red', 'winequality-white', 'yeast']
n_datasets = len(datasets)
for idx, dataset in enumerate(datasets):
    dataset = pd.read_csv("datasets/%s.dat" % (dataset), delimiter=",")
    y = dataset.iloc[:, -1:]
    columnNameClass = y.columns.values
    X = dataset.iloc[:, :-1]
    columnNames = X.columns.values
    X.columns = [''] * len(X.columns)
    y.columns = [''] * len(y.columns)
    # y.to_csv("Outputs\\{}.txt".format(idx), sep='\t', index=None)
    print(X)
