# -*- coding: utf-8 -*-
"""
Created on Sun May 14 07:35:29 2017

@author: Cloud
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import sklearn.linear_model as lin
import sklearn.naive_bayes as nb
import sklearn.model_selection as mod
import sklearn.metrics as met
import sklearn.preprocessing as pp
import scipy.stats as st

# get only columns that indicate heroes being used
def filterColumns(data):
    cols = ['radiant_win']
    for i in range(115):
        if i != 24:
            cols.append(str(i))

    return ds.filter(items=cols)

def preprocess(data):
    # split the data between allies and opponents
    r, c = data.shape
    opponents = np.zeros((r,c))
    cond = np.where(data == -1)
    for i,j in zip(cond[0], cond[1]):
        opponents[i,j] = 1
        data[i,j] = 0
    newData = np.hstack((data, opponents))
    return newData
    
def findBestParam(X, y, model, params):
    clf = mod.GridSearchCV(model(), params, cv=10)
    clf.fit(X, y)
    return clf.best_params_, clf.best_score_

def fitPredict(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = met.accuracy_score(y_test, y_pred)
    precision = met.precision_score(y_test, y_pred)
    recall = met.recall_score(y_test, y_pred)
    f1score = met.f1_score(y_test, y_pred) 
    return accuracy, precision, recall, f1score

def crossValidation(X, y, models, folds = 10):
    scores = np.zeros((len(models), folds, 4))
    for i in range(folds):
        X_train, X_test, y_train, y_test = mod.train_test_split(X, y, test_size=1/folds)    
        for model, m in zip(models, range(len(models))):
            scores[m,i] = fitPredict(X_train, y_train, X_test, y_test, model)
        
    return scores
    
def bootstrap_ci(stats, level = 0.05):
    lower_percentile = 0.5*level
    upper_percentile = 100 - 0.5*level
    return np.percentile(stats, lower_percentile), np.percentile(stats, upper_percentile)

def subsample(X, y, sample_size):
    xy_tuples = list(zip(X, y))
    xy_sample = [random.choice(xy_tuples) for _ in range(sample_size)]
    X_sample, y_sample = zip(*xy_sample)
    return X_sample, y_sample

def bootstrap(X, y, model, numSamples = 100, folds = 10):
    bs_results = np.zeros((numSamples,4))
    X_train, X_test, y_train, y_test = mod.train_test_split(X, y, test_size=1/folds)
    for i in range(numSamples):
        X_sample, y_sample = subsample(X_train, y_train, len(y_train))
        bs_results[i] = fitPredict(X_sample, y_sample, X_test, y_test, model)
    
    return bs_results
 
def plotErrorCurve(X, y, model, folds = 10):
    X_train, X_test, y_train, y_test = mod.train_test_split(X, y, test_size=1/folds)
    data_sizes = []
    train_errors = []
    test_errors = []
    for i in range(folds):
        size = int((i+1) / folds * len(y_train))
        X_sample, y_sample = subsample(X_train, y_train, size)
        model.fit(X_sample, y_sample)
        data_sizes.append(size)
        train_errors.append(1 - model.score(X_sample, y_sample))
        test_errors.append(1 - model.score(X_test, y_test))
    pl.plot(data_sizes, train_errors, c='b', label='Training error')
    pl.plot(data_sizes, test_errors, c='r', label='Generalisation error')
    pl.ylim(0,1)
    pl.ylabel('Error')
    pl.xlabel('Number of training samples')
    pl.title(model.__class__.__name__)
    pl.legend()
    pl.show()

def mcnemar(x, y):
    n1 = np.sum(x < y)
    n2 = np.sum(x > y)
    stat = np.power(n1-n2,2) / (n1+n2)
    pval = st.chi2.sf(stat,1)
    return stat, pval

def ttest_across_folds(model1scores, model2scores):
    return st.ttest_rel(model1scores, model2scores).pvalue * 0.5

if __name__ == '__main__':
    # read data
    ds = pd.read_csv('dota2data.csv')
    filteredDs = filterColumns(ds)
    X = filteredDs.iloc[:,1:].values
    y = filteredDs['radiant_win'].values
    # preprocessing
    Xpc = preprocess(X)
    X_train, X_test, y_train, y_test = mod.train_test_split(Xpc, y, test_size=0.1)
    
    # find best parameters 
    lr_parameters = [{'penalty':['l2'], 'solver':['lbfgs', 'newton-cg'], 'C':[0.01,0.1,1,10,100]}, {'penalty':['l1'], 'solver':['liblinear'], 'C':[0.01,0.1,1,10,100]}]
    bestLRParams, bestLRScore = findBestParam(X_train, y_train, lin.LogisticRegression, lr_parameters)    
    print(bestLRParams, bestLRScore)   
    lr = lin.LogisticRegression(penalty = bestLRParams['penalty'], solver = bestLRParams['solver'], C = bestLRParams['C'])
    
    # compare with naive bayes as a baseline model
    nb = nb.BernoulliNB()
    models = [lr, nb]
    scores = crossValidation(X_train, y_train, models)
    
    # plot error for model LR
    plotErrorCurve(X_train, y_train, lr)
    
    # signifance test
    ttest = ttest_across_folds(scores[0],scores[1])
    print("T-test score: {}".format(ttest))

    # McNemar's test  
    lr.fit(X_train, y_train)
    lrYPred = lr.predict(X_test)
    nb.fit(X_train, y_train)
    nbYPred = nb.predict(X_test)
    
    lr_yn = y_test == lrYPred
    nb_yn = y_test == nbYPred
    print("Logistic Regression accuracy: {}".format(np.sum(lr_yn) / len(y_test)))
    print("Naive Bayes accuracy: {}".format(np.sum(nb_yn) / len(y_test)))
    cmp = mcnemar(lr_yn,nb_yn)
    print("McNemar's test score: {}".format(cmp))

    # plot the lr confusion matrix
    pl.figure()
    pl.matshow(met.confusion_matrix(y_test, lrYPred), interpolation='nearest')
    pl.colorbar()
    pl.ylabel('true label')
    pl.xlabel('predicted label')
    pl.title("Logistic Regression Confusion Matrix")
    pl.show()
    
    #find CI for model LR 
    bsScores = bootstrap(Xpc,y,lr)
    meanScores = np.mean(bsScores, axis = 0)
    criteria = ['accuracy', 'precision', 'recall', 'f1-score']
    for c in range(4):
        lb, ub = bootstrap_ci(bsScores[:,c])
        print("Logistic Regression {} CI: {} [{},{}]".format(criteria[c],meanScores[c], lb, ub))
        