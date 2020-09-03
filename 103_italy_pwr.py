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
    print('Shape of X, y train and test dataset',X_train.shape, y_train.shape, X_test.shape, y_test.shape, end='\n')
    print('X_train:', X_train.head(), end='\n')
    print(X_train.info(), end='\n')
    
    labels, counts = np.unique(y_train, return_counts=True)
    print('\nThere are',labels, 'labels in this dataset. The counter of each one is', counts, end='\n')
    
    """
    #2. Plotting the classification labels
    fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
    for label in labels:
        X_train.loc[y_train == label, "dim_0"].iloc[0].plot(ax=ax, label=f"class {label}")
    plt.legend()
    ax.set(title="Example time series", xlabel="Time");
    plt.show()
    """
    #3. Creating a Model, Fit and Predict Sklearn Classifier
    #3.1.1 Sktime Tabularizing the data
    X_train_tab = tabularize(X_train)
    X_test_tab = tabularize(X_test)
    print('\n X_train tabularized\n',X_train_tab.head(), end='\n')

    #3.1.2 SKlearn RandomForest Classifier
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train_tab, y_train)
    y_pred = classifier.predict(X_test_tab)
    print('Accuracy sklearn RandomForestClassifier',round(accuracy_score(y_test, y_pred),4), end='\n')
    
    #3.2.1 Same SKlearn as above but using make_pipeline w/ Sktime Tabularizer 
    classifier = make_pipeline(Tabularizer(), RandomForestClassifier(n_estimators=100), verbose=True)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test), end='\n')
    
    sys.exit()
    #4. Feature extraction from the Time-series
    transformer = TSFreshFeatureExtractor(default_fc_parameters="minimal")
    extracted_features = transformer.fit_transform(X_train)
    print(extracted_features.head(), end='\n')
    
    #4.1 Sklearn using make_pipeline w/ Sktime TSFreshFeatureExtractor
    classifier = make_pipeline(TSFreshFeatureExtractor(show_warnings=False), RandomForestClassifier(n_estimators=100))
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test), end='\n')

    #5. Using Time series algorithms and classifiers from sklearn/sktime
    #5.1 
    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals='sqrt')), #Sktime
        ('transform', FeatureUnion([ #Sklearn
            ('mean', RowTransformer(FunctionTransformer(func=np.mean, validate=False))), #sktime
            ('std', RowTransformer(FunctionTransformer(func=np.std, validate=False))), #sktime
            ('slope', RowTransformer(FunctionTransformer(func=time_series_slope, validate=False))) #sktime
        ])),
        ('clf', DecisionTreeClassifier()) #From Sklearn
    ]
    time_series_tree = Pipeline(steps) #sklearn 
    time_series_tree.fit(X_train, y_train)
    print(time_series_tree.score(X_test, y_test))

    #6. Using Time series Sktime
    tsf = TimeSeriesForestClassifier(
        estimator=time_series_tree,
        n_estimators=100,
        criterion='entropy',
        bootstrap=True,
        oob_score=True,
        random_state=1
    )

    tsf.fit(X_train, y_train)
    print(tsf.score(X_test, y_test))


if __name__ == '__main__':
    main()
