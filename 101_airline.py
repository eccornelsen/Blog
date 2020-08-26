import numpy as np
import matplotlib.pyplot as plt

import sktime
from sktime import datasets
from sktime.utils.plotting.forecasting import plot_ys
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split

def main():
    df = datasets.load_airline() #Univariate, monthly records from 1949 to 60 (144 records)
    y_train, y_test = temporal_train_test_split(df, test_size=36) #36 months for testing

    forecaster = NaiveForecaster(strategy='last') #model
    forecaster.fit(y_train) #fit
    fh = np.arange(1,len(y_test)+1) #forecast horizon
    y_pred = forecaster.predict(fh) #pred

    plot_ys(y_train, y_test, y_pred, labels=['Train','Test','Naive Forecaster'])
    plt.xlabel('Months')
    plt.ylabel('Number of flights')
    plt.title('Time series of the number of international flights in function of time')
    plt.show()


    

if __name__== '__main__':
    main()
