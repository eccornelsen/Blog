import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_italy_power_demand
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.reduce import Tabularizer
from sktime.transformers.series_as_features.segment import RandomIntervalSegmenter
from sktime.transformers.series_as_features.summarize import TSFreshFeatureExtractor
from sktime.transformers.series_as_features.summarize import RandomIntervalFeatureExtractor
from sktime.utils.data_container import tabularize
from sktime.utils.time_series import time_series_slope

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf

def main():
    #1. Loading and splitting the dataset
    X_train, y_train = load_italy_power_demand(split='train', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='test', return_X_y=True)
    print('Shape of X, y train and test dataset',X_train.shape, y_train.shape, X_test.shape, y_test.shape, '\n')
    print('X_train:', X_train.head(), '\n')
    print('\nX_train info', X_train.info(), '\n')
    
    labels, counts = np.unique(y_train, return_counts=True)
    print('\nThere are',labels, 'labels in this dataset, one corresponds to winter and the other to summer. The counter of each one is', counts, '\n')
    
    #2. Creating a Model, Fit and Predict Sklearn Classifier
    #Sktime Tabularizing the data
    X_train_tab = tabularize(X_train)
    X_test_tab = tabularize(X_test)
    print('\n X_train tabularized\n',X_train_tab.head(), '\n')

    #2.1 SKlearn RandomForest Classifier
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train_tab, y_train)
    y_pred = classifier.predict(X_test_tab)
    print('Accuracy sklearn RandomForestClassifier',round(accuracy_score(y_test, y_pred),4), '\n')
    
    #2.2 Same SKlearn as above but using make_pipeline w/ Sktime Tabularizer 
    classifier = make_pipeline(Tabularizer(), RandomForestClassifier(n_estimators=100), verbose=True)
    classifier.fit(X_train, y_train)
    print('Accuracy sklearn RandomForestClassifier using sklearn make_pipeline in which the first step is to sktime Tabularize()', round(classifier.score(X_test, y_test),4), '\n')
     
    #3 Sklearn using make_pipeline w/ Sktime TSFreshFeatureExtractor
    classifier = make_pipeline(TSFreshFeatureExtractor(show_warnings=False), RandomForestClassifier(n_estimators=100))
    classifier.fit(X_train, y_train)
    print('Accuracy sklearn RandomForestClassifier using sklearn make_pipeline in which the first step is to sktime TSFreshFeatureExtractor that automatically extracts and filters several key statistical features from the nested X_train time series', round(classifier.score(X_test, y_test),4), '\n')
    
    #4. Using Time series algorithms and classifiers from sklearn/sktime 
    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals='sqrt')), #Sktime
        ('transform', FeatureUnion([ #Sklearn
            ('mean', RowTransformer(FunctionTransformer(func=np.mean, validate=False))), #sktime
            ('std', RowTransformer(FunctionTransformer(func=np.std, validate=False))), #sktime
            ('slope', RowTransformer(FunctionTransformer(func=time_series_slope, validate=False))) #sktime
        ])),
        ('clf', DecisionTreeClassifier()) #From Sklearn
    ]
    time_series_tree = Pipeline(steps, verbose=True) #sklearn 
    time_series_tree.fit(X_train, y_train)
    print('Accuracy sklearn DecisionTreeClassifier using sklearn Pipeline() as well as segmentation and transformation techniques from sktime and sklearn', round(time_series_tree.score(X_test, y_test),4))
    
    #5. Using Time series Sktime
    tsf = TimeSeriesForestClassifier(
        n_estimators=100,
        verbose=True
    )
    tsf.fit(X_train, y_train)
    print('Accuracy sktime TimeSeriesForestClassifier',round(tsf.score(X_test, y_test),4))

if __name__ == '__main__':
    main()
