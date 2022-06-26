#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 03:15:39 2022

@author: sunilguglani
"""


import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date
from h2o.automl import H2OAutoML
import h2o
from h2o.estimators import H2OXGBoostEstimator

try:
    h2o.cluster().shutdown()
except:
    pass
h2o.init(min_mem_size_GB=8)

filepath='/Users/sunilguglani/opt/anaconda3/lib/Algos/TradingStrategies/DL/'
filename_algo_2='icici_stocks_positional_directional_models_list.csv'
filename_algo_2_automl='icici_stocks_positional_directional_models_list_automl.csv'



class stocks_positional_directional(object):
    def __init__(self,script_predicted ):
        self.script_predicted=script_predicted 
        
        
        self.training_date,self.prediction_variable=pd.to_datetime(date(2022,1,1)),'^NSE_D_Close_pct_change_n_1'
        self.numb_days_of_data_required_train,self.numb_days_of_data_required_predict=4000,500 #int(df_params_temp['numb_days_of_data_required_train'].iloc[0]),int(df_params_temp['numb_days_of_data_required_predict'].iloc[0])
	 
    def convert_to_string(self,list): 
        s=''
        for i in list:
            s=s+str(i)+','
        
        s=s+'\n'
        return str(s)
    
    def train_xg(self,h2o_df,df,x,y):
        
        predictors = x
        response = y
        train,test=h2o_df.split_frame(ratios=[0.8])
    
        # Build and train the model:
        stocks_xgb = H2OXGBoostEstimator(booster='dart',
                                          normalize_type="tree",
                                          seed=1234)
        
        stocks_xgb.train(x=predictors,
                          y=response,
                          training_frame=train,
                          validation_frame=test)
        perf = stocks_xgb.model_performance()
        print(perf)

        path="/Applications/anaconda3/lib/Algos/TradingStrategies/DL/testing/"
    
        model_path = stocks_xgb.download_mojo( 
            path=path)
        print(model_path)
    
        stocks_xgb = h2o.upload_mojo(model_path)
    
        pred = stocks_xgb.predict(test)
        test_df=test.as_data_frame()
        
        print('mse:',stocks_xgb.mse(valid=True))
        print('mae:',stocks_xgb.mae(valid=True))
        try:
            ra_plot = stocks_xgb.varimp_plot()
        except:
            pass
        predict_list=h2o.as_list(pred, use_pandas=False)
    
        predict_list.remove(['predict'])
            
        df_predict=pd.DataFrame(predict_list)
        
        test_df['predict']=df_predict[0].values
        
        test_df['predict']=pd.to_numeric(test_df['predict']*1)
        test_df['weight']=np.abs(test_df['predict']*1)
        
        test_df['predict_dir']=np.sign(test_df['predict'])
            
        test_df['returns']=test_df['predict_dir']*test_df[y]
        test_df['accuracy']=(np.sign(test_df[y])==test_df['predict_dir'])
        
        accuracy_for_0_weight=round(float(test_df[test_df['accuracy']==True]['accuracy'].count()/test_df['accuracy'].count()),2)
        returns_for_0_weight=test_df['returns'].sum()
        count_for_0_weight=test_df['returns'].count()
        
        weight_filter=(test_df.weight>0.25)
        accuracy_for_04_weight=round(float(test_df[(test_df['accuracy']==True)&weight_filter]['accuracy'].count()/test_df[weight_filter]['accuracy'].count()),2)
        returns_for_04_weight=test_df[weight_filter]['returns'].sum()
        count_for_04_weight=test_df[weight_filter]['returns'].count()
        
        print('accuracy_for_0_weight,returns_for_0_weight',accuracy_for_0_weight,returns_for_0_weight)
        print('accuracy_for_04_weight,returns_for_04_weight',accuracy_for_04_weight,returns_for_04_weight)
        lst=[str(datetime.now()),self.script_predicted, model_path,accuracy_for_0_weight,returns_for_0_weight,count_for_0_weight,accuracy_for_04_weight,returns_for_04_weight,count_for_04_weight]
        file_write=self.convert_to_string(lst)
        with open(filepath+filename_algo_2, 'a') as file:
            file.write(file_write)


        # Extract feature interactions:
        #feature_interactions = stocks_xgb.feature_interaction()
        return test_df,model_path,accuracy_for_0_weight,returns_for_0_weight,count_for_0_weight,accuracy_for_04_weight,returns_for_04_weight,count_for_04_weight

    def train_automl(self,h2o_df,df,x,y):
        
        predictors = x
        response = y
        train,test=h2o_df.split_frame(ratios=[0.8])
    
        aml = H2OAutoML(max_models=20)

        aml.train(x=predictors,
                          y=response,
                          training_frame=train)
        
        best_model = aml.leader
        print(best_model)

        lb = aml.leaderboard
        lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
        print(lb)
        perf=best_model.model_performance(test)

        path="/Applications/anaconda3/lib/Algos/TradingStrategies/DL/testing/"
    
        model_path = best_model.download_mojo( 
            path=path)
        print(model_path)
        #model_path= '/Users/sunilguglani/opt/anaconda3/lib/Algos/TradingStrategies/DL/testing/XGBoost_model_python_1622570774923_107.zip'
        #model_path= '/Applications/anaconda3/lib/Algos/TradingStrategies/DL/testing/XGBoost_model_python_1651225180530_1.zip'
    
        best_model = h2o.upload_mojo(model_path)
    
        pred = best_model.predict(test)
        test_df=test.as_data_frame()
        
        print('mse:',best_model.mse(valid=True))
        print('mae:',best_model.mae(valid=True))
        try:
            ra_plot = best_model.varimp_plot()
        except:
            pass
        predict_list=h2o.as_list(pred, use_pandas=False)
    
        predict_list.remove(['predict'])
            
        df_predict=pd.DataFrame(predict_list)
        
        test_df['predict']=df_predict[0].values
        
        test_df['predict']=pd.to_numeric(test_df['predict']*1)
        test_df['weight']=np.abs(test_df['predict']*1)
        
        test_df['predict_dir']=np.sign(test_df['predict'])
            
        test_df['returns']=test_df['predict_dir']*test_df[y]
        test_df['accuracy']=(np.sign(test_df[y])==test_df['predict_dir'])
        
        accuracy_for_0_weight=round(float(test_df[test_df['accuracy']==True]['accuracy'].count()/test_df['accuracy'].count()),2)
        returns_for_0_weight=test_df['returns'].sum()
        count_for_0_weight=test_df['returns'].count()
        
        weight_filter=(test_df.weight>0.25)
        accuracy_for_04_weight=round(float(test_df[(test_df['accuracy']==True)&weight_filter]['accuracy'].count()/test_df[weight_filter]['accuracy'].count()),2)
        returns_for_04_weight=test_df[weight_filter]['returns'].sum()
        count_for_04_weight=test_df[weight_filter]['returns'].count()
        
        print('accuracy_for_0_weight,returns_for_0_weight',accuracy_for_0_weight,returns_for_0_weight)
        print('accuracy_for_04_weight,returns_for_04_weight',accuracy_for_04_weight,returns_for_04_weight)
        lst=[str(datetime.now()),self.script_predicted, model_path,accuracy_for_0_weight,returns_for_0_weight,count_for_0_weight,accuracy_for_04_weight,returns_for_04_weight,count_for_04_weight]
        file_write=self.convert_to_string(lst)
        with open(filepath+filename_algo_2_automl, 'a') as file:
            file.write(file_write)


        # Extract feature interactions:
        #feature_interactions = stocks_xgb.feature_interaction()
        return test_df,model_path,accuracy_for_0_weight,returns_for_0_weight,count_for_0_weight,accuracy_for_04_weight,returns_for_04_weight,count_for_04_weight,lb
    

    def validate(self,h2o_df,df,model_path,x,y):
        
        predictors = x
        response = y
         
        
        stocks_xgb = h2o.upload_mojo(model_path)
    
        pred = stocks_xgb.predict(h2o_df)
        test_df=df
    
        predict_list=h2o.as_list(pred, use_pandas=False)
    
    
        predict_list.remove(['predict'])
        
        
        df_predict=pd.DataFrame(predict_list)
        
        
        test_df['predict']=df_predict[0].values
        
        test_df['predict']=pd.to_numeric(test_df['predict']*1)
        test_df['weight']=np.abs(test_df['predict']*1)
        
        test_df['predict_dir']=np.sign(test_df['predict'])
    
        test_df['returns']=test_df['predict_dir']*test_df[y]
        test_df['accuracy']=(np.sign(test_df[y])==test_df['predict_dir'])

        accuracy_for_0_weight=round(float(test_df[test_df['accuracy']==True]['accuracy'].count()/test_df['accuracy'].count()),2)
        returns_for_0_weight=test_df['returns'].sum()
        count_for_0_weight=test_df['returns'].count()

        weight_filter=(test_df.weight>0.25)
        accuracy_for_04_weight=round(float(test_df[(test_df['accuracy']==True)&weight_filter]['accuracy'].count()/test_df[weight_filter]['accuracy'].count()),2)
        returns_for_04_weight=test_df[weight_filter]['returns'].sum()
        count_for_04_weight=test_df[weight_filter]['returns'].count()
        
        print('accuracy_for_0_weight,returns_for_0_weight',accuracy_for_0_weight,returns_for_0_weight)
        print('accuracy_for_04_weight,returns_for_04_weight',accuracy_for_04_weight,returns_for_04_weight)

        # Extract feature interactions:
        #feature_interactions = stocks_xgb.feature_interaction()
        return test_df,accuracy_for_0_weight,returns_for_0_weight,count_for_0_weight,accuracy_for_04_weight,returns_for_04_weight,count_for_04_weight
        
        
        accuracy_for_0_weight=round(float(test_df[test_df['accuracy']==True]['accuracy'].count()/test_df['accuracy'].count()),2)
        returns_for_0_weight=test_df['returns'].sum()
        count_for_0_weight=test_df['returns'].count()

                    
        # Extract feature interactions:
        #feature_interactions = stocks_xgb.feature_interaction()
        return test_df 
    
    def data_preprocessing(self,instrument_name,numb_days_of_data_required,baseline_date,mode='dev'):
        df=pd.DataFrame()

        #numb_days_of_data_required=100#365*3.5
        #numb_days_of_data_required=365*4.5
        if mode=='dev':
            numb_days_of_data_required=self.numb_days_of_data_required_train
        else:
            numb_days_of_data_required=self.numb_days_of_data_required_predict
        

        to_period=datetime.now().date()
        from_period=to_period-timedelta(days=numb_days_of_data_required)
        
        df_nifty_10m=yf.download(instrument_name, period=str(numb_days_of_data_required)+'d',interval='1d')
        print('len of dataset',len(df_nifty_10m))
        t=df_nifty_10m.index
        df_nifty_10m.index=t.tz_localize(None)

        df_nifty_process_d_market_data= self.process_nifty_d(df_nifty_10m)
        df_10m=df_nifty_process_d_market_data.copy()
        
        t=df_nifty_10m.index
        df_nifty_10m.index=t.tz_localize(None)
        self.df=df_10m
        df=df_10m.copy(deep=True)
        df.dropna(axis=1,how='all',inplace=True)
        df.dropna(axis=0,how='all',inplace=True)
        df=df[120:len(df)].copy()
        df.fillna(method='ffill',inplace=True)

        if mode=='dev':    
            df=df[np.abs(df[self.prediction_variable])>0].copy(deep=True)
            df.dropna(inplace=True)

            df=df[df.index<=baseline_date]
        if mode=='live':
            df=df[df.index>baseline_date]
            df_nifty_10m=df_nifty_10m[df_nifty_10m.index>baseline_date]
        
        list_cols=list(self.df.columns)

        y=self.prediction_variable
        print('length of the df',len(df))
        list_cols.remove(y)
        x=list_cols

        h2o_df=h2o.H2OFrame(df)

        return df,h2o_df,df_nifty_10m,x,y
    
   
    def process_nifty_d(self,df_nifty):
        scriptb='^NSE_D'
        
        df_nifty['Open_pct_change']=df_nifty['Open'].pct_change()*100
        df_nifty['Close_pct_change']=df_nifty['Close'].pct_change()*100
        
        df_nifty['High_pct_change']=df_nifty['High'].pct_change()*100
        df_nifty['Low_pct_change']=df_nifty['Low'].pct_change()*100

        df_nifty['Close_pct_change_n_1']=df_nifty['Close_pct_change'].shift(-1)
        
        df_nifty['Open_pct_change_p_1']=df_nifty['Open_pct_change'].shift(1)
        df_nifty['Close_pct_change_p_1']=df_nifty['Close_pct_change'].shift(1)
        df_nifty['High_pct_change_p_1']=df_nifty['High_pct_change'].shift(1)
        df_nifty['Low_pct_change_p_1']=df_nifty['Low_pct_change'].shift(1)

        df_nifty['Open_pct_change_p_2']=df_nifty['Open_pct_change'].shift(2)
        df_nifty['Close_pct_change_p_2']=df_nifty['Close_pct_change'].shift(2)
        df_nifty['High_pct_change_p_2']=df_nifty['High_pct_change'].shift(2)
        df_nifty['Low_pct_change_p_2']=df_nifty['Low_pct_change'].shift(2)
        
        df_nifty['Open_PrevOpen_pct_change']=(df_nifty['Open']-df_nifty['Open'].shift(1))*100/df_nifty['Open']
        df_nifty['Open_PrevClose_pct_change']=(df_nifty['Open']-df_nifty['Close'].shift(1))*100/df_nifty['Open']
        df_nifty['Open_PrevHigh_pct_change']=(df_nifty['Open']-df_nifty['High'].shift(1))*100/df_nifty['Open']
        df_nifty['Open_PrevLow_pct_change']=(df_nifty['Open']-df_nifty['Low'].shift(1))*100/df_nifty['Open']
    
        df_nifty['Open_High_pct_change']=(df_nifty['Open']-df_nifty['High'].shift(0))*100/df_nifty['Open']

        df_nifty['SMA']=df_nifty['Close'].rolling(12).mean()
    
        df_nifty['SMA_ratio']=df_nifty['SMA']/df_nifty['Close']
    
        df_nifty.index=pd.to_datetime(df_nifty.index)
        df_nifty['weekday']=df_nifty.index.weekday/7
        df_nifty=df_nifty[['Open_pct_change','High_pct_change','Low_pct_change','Close_pct_change','Close_pct_change_n_1',
                            'Open_PrevOpen_pct_change','Open_PrevClose_pct_change'
                           ,'Open_PrevHigh_pct_change','Open_PrevLow_pct_change'
                           ,'weekday','Open_pct_change_p_1'
                           ,'Close_pct_change_p_1','High_pct_change_p_1'
                           ,'Low_pct_change_p_1',
                           'Open_pct_change_p_2'
                           ,'Close_pct_change_p_2','High_pct_change_p_2'
                           ,'Low_pct_change_p_2','SMA_ratio'
                           ]]
                           
        for col in df_nifty.columns:
            df_nifty.rename({col:(scriptb+'_'+col)},axis=1,inplace=True)
    
        return df_nifty
    
list_scripts=[	'ADANIPORTS',	'AMBUJACEM',	'ASIANPAINT',	'AUROPHARMA','AXISBANK',	'BAJAJ-AUTO',	'BAJFINANCE',	'BOSCHLTD',	'BPCL',	'CIPLA',	'COALINDIA',	'DRREDDY',	'EICHERMOT',	'GAIL',	'HCLTECH',	'HDFC',	'HDFCBANK',	'HEROMOTOCO',	'HINDALCO',	'HINDPETRO',	'HINDUNILVR',	'ICICIBANK',	'INDUSINDBK',	'INFY',	'IOC',	'ITC',	'LT',	'LUPIN',	'M&M',	'MARUTI',	'NTPC',	'ONGC',	'POWERGRID',	'RELIANCE',	'SBIN',	'SUNPHARMA',	'TATASTEEL',	'TCS',	'TECHM',	'ULTRACEMCO',	'UPL',	'VEDL',	'WIPRO',	'ZEEL']


def train_models(script_local,automl=True):
    #script_local='ITC.NS'
    #script_local='^NSEBANK'
    #script_local='^NSEI'

    obj= stocks_positional_directional(script_local)
    df,h2o_df,df_nifty_10m,x,y=obj.data_preprocessing(instrument_name=script_local, numb_days_of_data_required=4000, baseline_date=obj.training_date,mode='dev')
    if automl==True:
        test_df,model_path,accuracy_for_0_weight,returns_for_0_weight,count_for_0_weight,accuracy_for_04_weight,returns_for_04_weight,count_for_04_weight,lb=obj.train_automl(h2o_df,df,x,y)
        print(lb)
    else:
        test_df,model_path,accuracy_for_0_weight,returns_for_0_weight,count_for_0_weight,accuracy_for_04_weight,returns_for_04_weight,count_for_04_weight=obj.train_xg(h2o_df,df,x,y)
    

def predict_models(script_local,minSegSize=1,automl=True):
    #script_local='ITC.NS'
    #script_local='^NSEBANK'

    obj= stocks_positional_directional(script_local)
    if automl==True:
        df_models=pd.read_csv(filepath+filename_algo_2_automl,names=['Datetime','script', 'model_path','accuracy_for_0_weight','returns_for_0_weight','count_for_0_weight','accuracy_for_04_weight','returns_for_04_weight','count_for_04_weight'],index_col=False)
    else:
        df_models=pd.read_csv(filepath+filename_algo_2,names=['Datetime','script', 'model_path','accuracy_for_0_weight','returns_for_0_weight','count_for_0_weight','accuracy_for_04_weight','returns_for_04_weight','count_for_04_weight'],index_col=False)
    
    df_models=df_models[df_models['script']==script_local]
    script=df_models['script'].iloc[-1]
    model_path=df_models['model_path'].iloc[-1]
    

    df,h2o_df,df_nifty_10m,x,y=obj.data_preprocessing(instrument_name=script_local, numb_days_of_data_required=600, baseline_date=obj.training_date,mode='live')

    test_df,accuracy_for_0_weight,returns_for_0_weight,count_for_0_weight,accuracy_for_04_weight,returns_for_04_weight,count_for_04_weight= obj.validate(h2o_df,df,model_path,x,y)
    test_df=pd.merge(test_df,df_nifty_10m,how='inner',left_index=True,right_index=True)
    test_df['predicted_price']=np.round(test_df['Close']*(100+test_df['predict'])/100,1)
    test_df['curr_price']=np.round(test_df['Close'],3)

    test_df['predict']=np.round(test_df['predict'],3)
    test_df['^NSE_D_Close_pct_change_n_1']=np.round(test_df['^NSE_D_Close_pct_change_n_1'],3)

    lst1=test_df.columns
    test_df['Change_in_predict_dir']=np.where((test_df['predict_dir']==test_df['predict_dir'].shift(1)),0,1)

    test_df2=test_df[['predict', 'weight',
       'predict_dir', 'returns', 'accuracy','Change_in_predict_dir','Close']]
    test_df2.to_csv('5min_bnifty_pred_v2.csv')
    
    message=test_df.iloc[-1][['predict','curr_price','predicted_price',
       'predict_dir','Change_in_predict_dir']].astype(str)
    
    predict=test_df['predict'].iloc[-1]
    curr_price=test_df['curr_price'].iloc[-1]
    predicted_price=test_df['predicted_price'].iloc[-1]
    predict_dir=test_df['predict_dir'].iloc[-1]
    
    test_df['returns']=test_df['predict_dir']*test_df['^NSE_D_Close_pct_change_n_1']
    test_df['returns'].cumsum().plot()
    test_df[test_df['weight']>0.25]['returns'].cumsum().plot()

    predict=str(predict)
    curr_price=str(curr_price)
    predicted_price=str(predicted_price)
    predict_dir=str(predict_dir)

    message='predict:'+predict+','+'curr_price:'+curr_price+','+'predicted_price:'+predicted_price+','+'predict_dir:'+predict_dir
    
    return message,test_df2

script_local='^NSEI'
automl=True
#train_models(script_local=script_local,automl=automl)
message,test_df2=predict_models(script_local=script_local,automl=automl)

