import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
X = np.loadtxt(open("../data/iris.csv", "r"), delimiter=',')

def test_loo(data, test_index):

    train_data = np.delete(data, test_index, axis=0)
    test_data = data[test_index, :]
    
    train1 = train_data[train_data[:, 0] == 1]
    train2 = train_data[train_data[:, 0] == 2]
    train3 = train_data[train_data[:, 0] == 3]

    mean1 = np.mean(train1[:, 1:], axis=0)
    mean2 = np.mean(train2[:, 1:], axis=0)
    mean3 = np.mean(train3[:, 1:], axis=0)
    cov1 = np.cov(train1[:, 1:], rowvar=0)
    cov2 = np.cov(train2[:, 1:], rowvar=0)
    cov3 = np.cov(train3[:, 1:], rowvar=0)

    dist1 = multivariate_normal(mean=mean1, cov=cov1)
    dist2 = multivariate_normal(mean=mean2, cov=cov2)
    dist3 = multivariate_normal(mean=mean3, cov=cov3)

    p1 = dist1.pdf(test_data[1:])
    p2 = dist2.pdf(test_data[1:])
    p3 = dist3.pdf(test_data[1:])

    index = np.argmax((p1, p2, p3)) + 1
    point = data[test_index, 0]


    correct = index == point
    return correct

def classify(data):
    ncorrect = 0
    ntotal = data.shape[0]
    for index in range(ntotal):
        ncorrect = ncorrect + test_loo(data, index)
    percent_correct = ncorrect * 100.0 / ntotal
    return percent_correct

print(X)
print(classify(X))