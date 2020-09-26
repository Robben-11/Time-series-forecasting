import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV

def train_test_perfornace(reg, grid, finance, train_X_sc, train_y_sc, test_X_sc, test_y_sc, 
                          train, test, y_sc, time_split): 


    def MAPE(true, pred): 

        y_true, y_pred = np.array(true), np.array(pred)

        return round(abs((y_true- y_pred)/y_true).mean() * 100,4)

    RCV = GridSearchCV(estimator=reg , 
                   param_grid= grid, 
                   scoring='neg_mean_squared_error', 
                   n_jobs=-1, 
                   cv=time_split, 
                   verbose=50)

    RCV.fit(train_X_sc,train_y_sc) 

    print('Best estimator {}'.format(RCV.best_estimator_))
    print('Best score of grid search {}'.format(RCV.best_score_))
    
    # trian data retrival
    temp = yf.download('FB', start='2015-07-07', end='2015-07-09')['Adj Close']
    lis = [temp['2015-07-07']] + list(finance[finance['Date'] >= '2015-07-08'][finance['Date'] <= '2019-08-02']['FB'])
    fb_t_1_tr = pd.Series(lis)
    fb_true_tr = finance[finance['Date'] >= '2015-07-08'][finance['Date'] <= '2019-08-05']['FB']
    
    # trian performnace evaluation 
    yhat_return_sc_tr = RCV.best_estimator_.predict(train_X_sc)
    yhat_return_tr =  y_sc.inverse_transform(yhat_return_sc_tr) 
    yhat_stock_tr = (yhat_return_tr + 1) * fb_t_1_tr
    print(' Collecting the training result...%')
    print('mean squared error {}'.format(mean_squared_error(fb_true_tr, yhat_stock_tr))) 
    print('mean absolute error {}'.format(mean_absolute_error(fb_true_tr, yhat_stock_tr))) 
    print('mean absolute percentage error(%) {}'.format(MAPE(fb_true_tr, yhat_stock_tr)))

    # test data reterival
    fb_t_1 = finance[finance['Date'] >= '2019-08-05'][finance['Date'] <= '2020-08-06']['FB']
    fb_true = finance[finance['Date'] >= '2019-08-06'][finance['Date'] <= '2020-08-07']['FB']

    # test performance evaluation
    yhat_return_sc = RCV.best_estimator_.predict(test_X_sc)
    yhat_return =  y_sc.inverse_transform(yhat_return_sc) 
    yhat_stock = (yhat_return + 1) * fb_t_1
    print(' Collecting the testing result...%')
    print('mean squared error {}'.format(mean_squared_error(fb_true, yhat_stock))) 
    print('mean absolute error {}'.format(mean_absolute_error(fb_true, yhat_stock))) 
    print('mean absolute percentage error(%) {}'.format(MAPE(fb_true, yhat_stock)))
    
    
    #test plot 
    yhat_stock.index = test.index
    fb_true.index = test.index
    print('Plot Pred vs True on test set')
    plt.figure(figsize=(12, 8))
    yhat_stock.plot(label = 'Pred')
    fb_true.plot(label = 'True')
    plt.legend()
    plt.grid()
    
    # Residulas of point forecasting
    plt.figure(figsize=(12, 8))
    (fb_true - yhat_stock).plot(label = 'Residuals')
    plt.legend()
    plt.grid()

    return yhat_stock, fb_true, RCV