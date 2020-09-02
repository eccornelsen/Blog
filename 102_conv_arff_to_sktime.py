from scipy.io import arff
import pandas as pd
from sktime.utils.data_container import is_nested_dataframe
from sktime.utils.data_container import detabularize
from sktime.utils.load_data import load_from_arff_to_dataframe
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier


def main():
    #Load arff file into Tuple of size 2. 
    #First element has the time-series data in arrays and Second element has the description of the attributes
    TRAIN = arff.loadarff('ItalyPowerDemand_TRAIN.arff')
    TEST = arff.loadarff('ItalyPowerDemand_TEST.arff')
    #Convert the data from the first Tuple elemento to a tabularized dataframe
    df_TRAIN = pd.DataFrame(TRAIN[0])
    df_TEST = pd.DataFrame(TEST[0])
    
    #Using sktime to handle the data
    print(df_TRAIN.head())
    print('\n Is nested the df above?', is_nested_dataframe(df_TRAIN), '\n')
    
    #Handling the datasets
    X_train = df_TRAIN.drop('target',axis=1)
    y_train = df_TRAIN['target'].astype(int)
    print(X_train.head(), y_train.head(), '\n')
    X_test = df_TEST.drop('target',axis=1)
    y_test = df_TEST['target'].astype(int)

    #Detabularizing and Nesting X_train, X_test
    X_train_detab = detabularize(X_train)
    X_test_detab = detabularize(X_test)
    print(X_train_detab.head())
    print('Is nested the detabularized df above?', is_nested_dataframe(X_train_detab), '\n')

    #The lines above could be simplified with the following method from sktime
    X, y = load_from_arff_to_dataframe('ItalyPowerDemand_TRAIN.arff')
    print(X_train_detab.head(), X.head(), type(y_train), type(y))
    
    #Classifier algorithm
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    knn.fit(X_train_detab, y_train)
    print('The score of the KNN classifier is:', round(knn.score(X_test_detab, y_test),4))
   
if __name__ == '__main__':
    main()
