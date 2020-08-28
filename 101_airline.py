import numpy as np
import matplotlib.pyplot as plt

import sktime
from sktime import datasets
from sktime.utils.plotting.forecasting import plot_ys
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA 
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.compose import EnsembleForecaster
from sktime.performance_metrics.forecasting import smape_loss
from sktime.performance_metrics.forecasting import mase_loss

def main():
    df = datasets.load_airline() #Univariate, monthly records from 1949 to 60 (144 records)
    y_train, y_test = temporal_train_test_split(df, test_size=36) #36 months for testing

    forecaster = NaiveForecaster(strategy='seasonal_last',sp=12) #model strategy: last, mean, seasonal_last. sp=12months (yearly season)
    forecaster.fit(y_train) #fit
    fh = np.arange(1,len(y_test)+1) #forecast horizon: array with the same lenght of y_test
    y_pred = forecaster.predict(fh) #pred

    
    forecaster2 = AutoARIMA(sp=12, suppress_warnings=True, trace=1)
    forecaster2.fit(y_train)
    y_pred2 = forecaster2.predict(fh)
    
    forecaster3 = ExponentialSmoothing(trend='add', damped='True', seasonal='multiplicative', sp=12)
    forecaster3.fit(y_train)
    y_pred3 = forecaster3.predict(fh)

    forecaster4 = ThetaForecaster(sp=12)
    forecaster4.fit(y_train)
    y_pred4 = forecaster4.predict(fh)

    forecaster5 = EnsembleForecaster([
        ('NaiveForecaster', NaiveForecaster(strategy='seasonal_last',sp=12)), 
        ('AutoARIMA', AutoARIMA(sp=12, suppress_warnings=True)),
        ('Exp Smoothing', ExponentialSmoothing(trend='add', damped='True', seasonal='multiplicative', sp=12)),
        ('Theta', ThetaForecaster(sp=12))])
    forecaster5.fit(y_train)
    y_pred5 = forecaster5.predict(fh)
   

    plot_ys(y_train, y_test, y_pred, y_pred2, y_pred3, y_pred4, y_pred5, labels=['Train','Test','Naive Forecaster','AutoARIMA','Exp Smoothing','Theta', 'Ensemble'])
    plt.xlabel('Months')
    plt.ylabel('Number of flights')
    plt.title('Time series of the number of international flights in function of time')
    plt.show()

    print('SMAPE Error for NaiveForecaster is:', 100*round(smape_loss(y_test, y_pred),3), '%')
    print('SMAPE Error for AutoARIMA is:', 100*round(smape_loss(y_test, y_pred2),3), '%')
    print('SMAPE Error for Exp Smoothing is:', 100*round(smape_loss(y_test, y_pred3),3), '%')
    print('SMAPE Error for Theta is:', 100*round(smape_loss(y_test, y_pred4),3), '%')
    print('SMAPE Error for Ensemble is:', 100*round(smape_loss(y_test, y_pred5),3), '%')

if __name__== '__main__':
    main()
