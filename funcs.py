import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

#from keras.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error as mse, mae, mape



################################### Metrics ###################################


def errors_out_sample(test, forc):
    errors = {}
    errors['MSE'] = mse(list(test), forc)
    errors['MAE'] = mae(list(test), forc)
    errors['RMSE'] = np.sqrt(mse(list(test), forc))
    return errors


################################### TS ###################################


def test_stationarity(timeseries):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    print('\n')    
    #Perform KPSS test:
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries)
    kpssoutput = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpssoutput['Critical Value (%s)'%key] = value
    print(kpssoutput)
    #Graph
    plt.plot(list(timeseries))
    plt.show()

def diff_series(timeseries):
    diff_s = timeseries - timeseries.shift()
    return diff_s[1:]

def correlogram(series):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(list(series), lags=25, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(list(series), lags=25, ax=ax2)


def time_line_creator(start_time, row_len):
    time = [1,2,3,4]*(row_len//4)
    t = 0
    for q in range(len(time)):
        str_q = str(start_time+t) + '_' + str(time[q])
        if time[q] == 4:
            t+=1
        time[q] = str_q
    return time


################################### Plotting ###################################


def plot_forecast(train, test, forc, bbox_loc=(1.4,1)):
    '''
    train and test must be pd.DataFrame; forc must be ARIMAResult.forecast object
    '''
    # creating df for forecast
    df_forc = pd.DataFrame(forc[0], index=test.index, columns=['Forecast'])
    df_forc['std'] = forc[1]
    df_forc['conf_int_5'] = forc[2].T[0]
    df_forc['conf_int_95'] = forc[2].T[1]
    # plotting
    plt.figure(figsize=(9,5))
    plt.plot(train, label='Train_Sample')
    plt.plot(test, label='Test_Sample')
    plt.plot(df_forc['Forecast'], 'g')
    plt.plot(df_forc['conf_int_5'], 'k--', label='Confidence interval')
    plt.plot(df_forc['conf_int_95'], 'k--', label='_nolegend_')
    plt.legend(bbox_to_anchor=bbox_loc, loc='best')
    plt.show()

def plot_forecast_out_only(test, forc, conf_int=True, bbox_loc=(1.5,1)):
    '''
    test must be pd.DataFrame; forc must be ARIMAResult.forecast object
    '''
    # creating df for forecast
    df_forc = pd.DataFrame(forc[0], index=test.index, columns=['Forecast'])
    df_forc['std'] = forc[1]
    df_forc['conf_int_5'] = forc[2].T[0]
    df_forc['conf_int_95'] = forc[2].T[1]
    # plotting
    plt.figure(figsize=(7,4))
    plt.plot(test, '-o', label='Test_Sample')
    plt.plot(df_forc['Forecast'], 'g-o')
    if conf_int == True:
        plt.plot(df_forc['conf_int_5'], 'k--', label='Confidence interval')
        plt.plot(df_forc['conf_int_95'], 'k--', label='_nolegend_')
    plt.legend(bbox_to_anchor=bbox_loc, loc='best')
    plt.show()

######### NN Plotting

def nn_plot_in_sample(nn_model, train_nn, bbox_loc=(1.5,1)):
    plt.plot(nn_model.predict(np.array(train_nn.iloc[:, :-1])), label='NN_Predict')
    plt.plot(np.array(train_nn.iloc[:, -1]), color='blue', label='Train_Sample')
    plt.legend(bbox_to_anchor=bbox_loc, loc='best')
    plt.show()


def nn_plot_out_sample(nn_forecast, test, bbox_loc=(1.5,1)):
    plt.plot(
        nn_forecast,
        '-o',
        label='NN_Forecast'
    )
    plt.plot(list(test), '-o', color='blue', label='Test_Sample')
    plt.legend(bbox_to_anchor=bbox_loc, loc='best')
    plt.show()

######### Hybrid Plotting

def hybrid_plot_forecast(forc_arima, nn_resid_forecast, test, bbox_loc=(1.5,1)):
    plt.plot(forc_arima[0] + np.array(nn_resid_forecast), '-o', label='Hybrid_Forecast')
    plt.plot(list(test), '-o', label='Test_sample')
    plt.legend(bbox_to_anchor=bbox_loc, loc='best')
    plt.show()


################################### NN ###################################


def create_df_for_nn(ts, n_lags):
    #create list of lists for df
    list_for_lags = []
    for i in range(n_lags, len(ts-1)):
        list_for_lags.append(ts[i-n_lags:i+1].tolist())
    #Cols and Indexes for df
    names_for_df = []
    for j in range(n_lags):
        names_for_df.append('x'+str(j))
    names_for_df.append('y')
    ind_for_df = ts.index[n_lags:]
    #creating df with cols and indexes
    df_for_nn = pd.DataFrame(list_for_lags)
    df_for_nn.columns = names_for_df
    df_for_nn.index = ind_for_df
    return df_for_nn

def nn_predict_series(nn_model, last_obs, lags_to_predict):
    '''
    last_obs - pandas df
    '''
    lags = len(last_obs.columns)
    predicted_obs = np.array(last_obs)[0].tolist() # list
    obs = np.array([predicted_obs]) # np.array
    for i in range(lags_to_predict):
        predicted_obs.append(nn_model.predict(obs)[0][0])
        obs = np.array([predicted_obs[-lags:]]) # np.array
    return predicted_obs[lags:]

def last_obs_creator_nn(train_nn, lags_for_nn):
    return pd.DataFrame(train_nn.iloc[len(train_nn)-1, 1:]).T


################################### Residuals for Hybrid ###################################


def create_df_for_nn_residuals(resid_series, n_lags):
    #create list of lists for df
    list_for_lags = []
    for i in range(n_lags, len(resid_series-1)):
        list_for_lags.append(resid_series[i-n_lags:i+1])
    #Cols for df
    names_for_df = []
    for j in range(n_lags):
        names_for_df.append('x'+str(j))
    names_for_df.append('y')
    #creating df with cols and indexes
    df_for_nn = pd.DataFrame(list_for_lags)
    df_for_nn.columns = names_for_df
    return df_for_nn

def get_resids_arima(train, model_arima, d=1):
    if d==1:
        train_start = 0
    elif d==2:
        train_start = 1
    residuals_in_sample = np.array(train[train_start:]) - model_arima.predict(end=len(train), typ='levels')
    return residuals_in_sample

def last_lags_resid_df_for_nn(resids, test, n_lags):
    return pd.DataFrame(resids[-len(test)-n_lags:-len(test)]).T


################################### Hybrid ###################################


def hybrid_predict_series(forc_arima, nn_resid_forecast):
    return forc_arima[0] + np.array(nn_resid_forecast)


################################### Scaling ###################################


def scaler(series):
    mean = series.mean()
    z = series.max() - series.min()
    scaled_series = series.copy()
    scaled_series = (scaled_series-mean)/z
    return scaled_series, mean, z

def scaler_back(nn_predict, mean, z): #  , mean=scaler(resids_arima)[1], z=scaler(resids_arima)[2]):
    nn_forecast = np.array(nn_predict)*z + mean
    return nn_forecast