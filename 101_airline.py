import numpy as np
import matplotlib.pyplot as plt

import sktime
from sktime import datasets
from sktime.utils.plotting.forecasting import plot_ys
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA 
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.performance_metrics.forecasting import smape_loss

def main():
    df = datasets.load_airline() #Univariate, monthly records from 1949 to 60 (144 records)
    y_train, y_test = temporal_train_test_split(df, test_size=36) #36 months for testing

    forecaster = NaiveForecaster(strategy='seasonal_last',sp=12) #model: last, mean, seasonal_last w/ sp=12
    forecaster.fit(y_train) #fit
    fh = np.arange(1,len(y_test)+1) #forecast horizon
    y_pred = forecaster.predict(fh) #pred

    forecaster2 = AutoARIMA(sp=12, suppress_warnings=True)
    forecaster2.fit(y_train)
    y_pred2 = forecaster2.predict(fh)

    forecaster3 = ExponentialSmoothing(trend='add', seasonal='multiplicative', sp=12)
    forecaster3.fit(y_train)
    y_pred3 = forecaster3.predict(fh)


    plot_ys(y_train, y_test, y_pred, y_pred2, y_pred3, labels=['Train','Test','Naive Forecaster','AutoARIMA','Exp Smoothing'])
    plt.xlabel('Months')
    plt.ylabel('Number of flights')
    plt.title('Time series of the number of international flights in function of time')
    plt.show()

    print('SMAPE loss for NaiveForecaster is:', smape_loss(y_test, y_pred))
    print('SMAPE loss for AutoARIMA is:', smape_loss(y_test, y_pred2))
    print('SMAPE loss for Exp Smoothing is:', smape_loss(y_test, y_pred3))

if __name__== '__main__':
    main()
