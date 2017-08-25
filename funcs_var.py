import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

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
    return(diff_s[1:])

def undiff(previous_df, diff_df):
    result = []
    previous = previous_df.tail(1).as_matrix()[0]
    current = previous.copy()
    for a in diff_df.as_matrix():
        current = current + a
        result.append(list(current))
        previous = current.copy()
    result = pd.DataFrame(result)
    result.columns = diff_df.columns
    result.index = diff_df.index
    return result

def scaler(df):
    return (df-df.mean())/df.std(), df.mean(), df.std()

def scaler_back(df, mean, std):
    return df*std+mean

### NEURAL NET ###

def nn_transform(train, n):
    y = train.shift(-n)
    ind = train.index.tolist()
    ind = ind[n:]
    col_names = []
    df_list = [y]
    for i in range(n):
        df = train.shift(-i).copy()#.dropna().reset_index(drop=True).copy()
        col_names = []
        for c in df.columns:
            col_names.append(c+'_x'+str(i))
        df.columns = col_names
        df_list.append(df)
        final_df = pd.concat(df_list, axis=1).dropna()
        final_df.index = ind
    return final_df

def predict_df_maker(train, lags):
    df_ = train.iloc[-lags:, :].reset_index(drop=True).copy()
    df_list = []
    for i in df_.index:
        df = pd.DataFrame(df_.iloc[i, :]).T.reset_index(drop=True)
        col_names=[]
        for c in df.columns:
            col_names.append(c+'_x'+str(i))
        df.columns = col_names
        df_list.append(df)
    return pd.concat(df_list, axis=1)

def nn_predict(nn_model, train, lags, steps):
    full_train = train.copy()
    for i in range(steps):
        df_pred = predict_df_maker(full_train, lags=lags)
        pred = nn_model.predict(np.array(df_pred))
        pred = pd.DataFrame(pred)
        pred.columns = full_train.columns
        full_train = full_train.append(pred, ignore_index=True)
    return full_train.iloc[-steps:, :].reset_index(drop=True)

### PLOTTING ###

def plot_compar(forcast, real):
    plt.plot(real, color='blue', label='Real Value')
    plt.plot(forcast, color='red', label='Forecasted Value')
    plt.legend(loc=0)
    plt.show()

def error(forecast, real):
    mse_ = mse(real, forecast)
    mae_ = mae(real, forecast)
    rmse_ = np.sqrt(mse(real, forecast))
    return {'mse':mse_, 'mae':mae_, 'rmse':rmse_}
