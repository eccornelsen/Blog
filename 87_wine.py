import sys
import time

from collections import OrderedDict
import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def printMetrics(model, X_train, X_test, y_train, y_test):
    t_start = time.time()
    model.fit(X_train, y_train)
    t_end = time.time()
    predictions = model.predict(X_test)
    target_names = ['class 0', 'class 1', 'class 2']
    print(model)
    print(classification_report(y_test, predictions, target_names=target_names))
    print('Overall accuracy:', round(accuracy_score(y_test, predictions),5))
    print('Computation time:', round((t_end-t_start),3), 'seconds\n')

def printFeatures(model, X_train):
    feat_values = list(model.best_estimator_.feature_importances_*100)
    feat_names = list(X_train.columns)
    feat_dct = {feat_names[i]: feat_values[i] for i in range(len(feat_names))} 
    ord_feat_dct = OrderedDict(sorted(feat_dct.items(), key=lambda t: t[1], reverse=True)) #t[1] sort by val
    for k, v in ord_feat_dct.items(): 
        print(k, ':', round(v,2))
    print()

def main():
    dct = load_wine()
    df = pd.DataFrame(data = dct['data'], columns=dct['feature_names'])
    df['class'] = dct['target']
    y = df['class']
    X = df.drop('class',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    model = RandomForestClassifier(verbose=0)
    printMetrics(model, X_train, X_test, y_train, y_test)

    params = {'n_estimators':[50,60,70], 
              'max_features':['auto','sqrt','log2'], 
              'max_depth': [3,5]}
    randCV = RandomizedSearchCV(model, 
                                param_distributions=params, 
                                cv=10, n_iter=10, verbose=0, n_jobs=1)
    printMetrics(randCV, X_train, X_test, y_train, y_test)
    print('Best parameters:', randCV.best_params_) #print(randCV.cv_results_)
    printFeatures(randCV, X_train)

    
    gridCV = GridSearchCV(model, 
                          param_grid=params, 
                          cv=10, verbose=0, n_jobs=1)
    printMetrics(gridCV, X_train, X_test, y_train, y_test)
    print('Best parameters:', gridCV.best_params_)
    printFeatures(gridCV, X_train)

if __name__ == '__main__':
    main()
