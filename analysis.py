import numpy as np
from scipy.stats import rankdata, ranksums
from tabulate import tabulate

scores = np.load('./results.npy')
mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

mean_ranks = np.mean(ranks, axis=0)
clfs = ['GNB PCA', 'GNB LDA', 'SVM PCA', 'SVM LDA', 'kNN PCA',
            'kNN LDA', 'CART PCA', 'CART LDA']

print("\nModels:\n", clfs)
print("\nMean ranks:\n", mean_ranks)

alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])



names_column = np.expand_dims(np.array(clfs), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, clfs, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, clfs, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)


advantage = np.zeros((len(clfs), len(clfs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), clfs)
print("\nAdvantage:\n", advantage_table)


significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), clfs)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)