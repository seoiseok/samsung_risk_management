# samsung_risk_management

##비정상 시계열 데이터 전처리 과정 
  import numpy as np
  import seaborn as sns
  import matplotlib.pyplot as plt
  import pandas as pd

  import statsmodels.api as sm
  from statsmodels.tsa.api import VAR
  from statsmodels.tsa.stattools import adfuller

  data=pd.read_excel('/content/drive/MyDrive/공모전/로그파일2.xlsx')
  df=data[['거리두기단계','누적확진자수','매출액']]
  df.index=data['Date']
  c1=['매출액']
  c2=['거리두기단계']
  c3=['누적확진자수']
  
###변수별 그래프 확인
  df[c1].plot(figsize=(12.2,6.4))
  df[c2].plot(figsize=(12.2,6.4))
  df[c3].plot(figsize=(12.2,6.4))
  
###ADF KPSS 검정 
  n_obs = 1
df_train, df_test = df[0:-n_obs], df[-n_obs:]

def adf_test(df):
  result= adfuller(df.values)
  print('ADF Statistics; %f' % result[0])
  print('p-value: %f'% result[1])
  print('Critical values:')
  for key, value in result[4].items():
    print('\t%s: %.3f' %(key, value))

print('ADF Test: 매출액')
adf_test(df_train['매출액'])
print('ADF Test: 거리두기단계')
adf_test(df_train['거리두기단계'])
print('ADF Test: 누적확진자수')
adf_test(df_train['누적확진자수'])

from statsmodels.tsa.stattools import kpss

def kpss_test(df):
  statistic, p_value, n_lags, critical_values =kpss(df.values)
  print(f'KPSS Statistis: {statistic}')
  print(f'p-value: {p_value}')
  print(f'num lags: {n_lags}')
  print('Critical Values:')
  for key, value in critical_values.items():
    print(f' {key}:{value}')

print('KPSS Test: 매출액')
kpss_test(df_train['매출액'])
print('KPSS Test: 거리두기단계')
kpss_test(df_train['거리두기단계'])
print('KPSS Test: 누적확진자수')
kpss_test(df_train['누적확진자수'])


## 보험료 예측 모델 생성 
  ! pip install neuralprophet
  
  ###패키지 임포트
  import pandas as pd
  from neuralprophet import NeuralProphet, set_log_level
  import plotly.express as px
  set_log_level("ERROR")
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  data = pd.read_excel('/content/drive/MyDrive/공모전/전처리결과.xlsx')
  
  ###시계열 및 종속변수 이름 변경
  data = data.rename(columns={"Date":"ds","매출액":"y"})
  
  x_col_list=['누적확진자수','거리두기단계'] # X변수 정의
  y_col_list=['y'] # Y변수 정의 

  X=data[x_col_list]
  Y=data[y_col_list]

  X_diff=X.diff()
  X_diff.columns=X.columns+'_diff'


  ###Y변수들에 대한 lag 데이터 생성 (2 lag 까지만)
  Y_diff=Y.diff()
  Y_diff.columns=Y.columns+'_diff'

  ###X데이터 결합
  X=pd.concat([X,X_diff],axis=1)
  ###Y데이터 결합
  Y=pd.concat([Y,Y_diff],axis=1)
  ###전체 데이터셋 생성
  data_pret=pd.concat([data['ds'][2:],X,Y],axis=1)
  data_pret=data_pret.reset_index(drop=True)
  data_pret=data_pret.dropna()
  
  ###독립변수 정의
  col_lst=data_pret.columns
  col_lst=col_lst.drop(['ds','y'])
  col_lst=list(col_lst)
  col_lst
  ###train test split
  cutoff = "2021-10-01" #데이터 분할 기준
  train = data_pret[data_pret['ds']<cutoff]
  test = data_pret[data_pret['ds']>=cutoff]
  
  m = NeuralProphet(
    n_forecasts=10, 
    n_changepoints=3,
    n_lags=1, #1시간 뒤 regressor 지연 반영
    yearly_seasonality=30,
    num_hidden_layers=16, #히든 레이어 수 설정
    d_hidden=256, #은닉층 뉴런 설정
    learning_rate=0.01, #학습률 설정
    batch_size=4, #배치 사이즈 설정
    epochs=200, #학습 횟수
    drop_missing= True )

  m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

  ###독립 변인(변수) 추가 및 정규화
  m = m.add_lagged_regressor(names=col_lst,normalize="minmax") 

  ###학습 수행
  metrics = m.fit(train, freq="M", validation_df=test, progress='plot')
  
  ###metric 확인
  print("SmoothL1Loss: ", metrics.SmoothL1Loss.tail(1).item())
  print("MAE(Train): ", metrics.MAE.tail(1).item())
  print("MAE(Test): ", metrics.MAE_val.tail(1).item())
  
  ###학습 선 그래프 생성
  px.line(metrics, y=['MAE', 'MAE_val'], width=800, height=400)
  
  ###1년간 데이터 확인
  forecast = m.predict(test)
  m = m.highlight_nth_step_ahead_of_each_forecast(1)
  fig = m.plot(forecast[-11:])
  
  ### 훈련결과 확인 
  future = m.make_future_dataframe(train, periods=15, n_historic_predictions=len(train))
  forecast = m.predict(future)
  fig_forecast = m.plot(forecast)
