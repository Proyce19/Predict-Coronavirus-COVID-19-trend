import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from fbprophet import Prophet


df = pd.read_csv('datasets/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)




df_date = df.groupby(["Date"])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
# df_country = df.groupby(["Country"])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
#
# print(df_date.tail())
# print(df_country.tail())
#
date_x_ticks = []
country_x_ticks = []
date_confirmed=[]
date_deaths=[]
date_recovered=[]
country_confirmed = []
country_deaths = []
country_recovered = []

for index, row in df_date.iterrows():

    date_x_ticks.append(row['Date'])
    date_confirmed.append(row['Confirmed'])
    date_deaths.append(row['Deaths'])
    date_recovered.append(row['Recovered'])

#
# for index, row in df_country.iterrows():
#
#     country_x_ticks.append(row['Country'])
#     country_confirmed.append(row['Confirmed'])
#     country_deaths.append(row['Deaths'])
#     country_recovered.append(row['Recovered'])
#
#
#
# # plt.xticks(np.arange(len(date_x_ticks[:10])), date_x_ticks[:10])
# # plt.xlabel('Dates')
# # plt.ylabel('Cases')
# # plt.plot(date_confirmed[:10], label='Confirmed', color='blue')
# # plt.plot(date_deaths[:10], label='Deaths', color='red')
# # plt.plot(date_recovered[:10], label='Recovered', color='green')
# # plt.title("Coronavirus cases in the world by date")
# # plt.legend()
# # plt.show()
#
#
#
# #plt.xticks(np.arange(len(date_x_ticks)), date_x_ticks)
# plt.xlabel('Dates')
# plt.ylabel('Cases')
# plt.plot(date_confirmed, label='Confirmed', color='blue')
# plt.plot(date_deaths, label='Deaths', color='red')
# plt.plot(date_recovered, label='Recovered', color='green')
# plt.title("Coronavirus cases in the world by date")
# plt.legend()
# plt.show()
#
# #
# # plt.xticks(np.arange(len(country_x_ticks[:5])), country_x_ticks[:5])
# # plt.xlabel('Countries')
# # plt.ylabel('Cases')
# # plt.bar(np.arange(len(country_confirmed[:5])), country_confirmed[:5], align='center', alpha=0.5, color='blue', label='Confirmed')
# # plt.bar(np.arange(len(country_deaths[:5])), country_deaths[:5], align='center', alpha=0.5, color='red', label='Deaths')
# # plt.bar(np.arange(len(country_recovered[:5])), country_recovered[:5], align='center', alpha=0.5, color='green', label='Recovered')
#
# plt.xlabel('Countries')
# plt.ylabel('Cases')
# plt.bar(np.arange(len(country_confirmed)), country_confirmed, align='center', alpha=0.5, color='blue', label='Confirmed')
# plt.bar(np.arange(len(country_deaths)), country_deaths, align='center', alpha=0.5, color='red', label='Deaths')
# plt.bar(np.arange(len(country_recovered)), country_recovered, align='center', alpha=0.5, color='green', label='Recovered')
# plt.title("Coronavirus cases in the world by country")
# plt.legend()
# plt.show()
# #
date_confirmed_prophet = df_date[['Date', 'Confirmed']]
date_death_prophet = df_date[['Date', 'Deaths']]
date_recovered_prophet = df_date[['Date', 'Recovered']]

date_confirmed_prophet.columns = ['ds', 'y']
date_death_prophet.columns = ['ds', 'y']
date_recovered_prophet.columns = ['ds', 'y']
#
model_confirmed = Prophet(interval_width=0.99)
model_confirmed.fit(date_confirmed_prophet)
future_confirmed = model_confirmed.make_future_dataframe(periods=30)
forecast_confirmed = model_confirmed.predict(future_confirmed)

print(forecast_confirmed.tail())
#

forecast_confirmed_yhat = []
forecast_confirmed_yhat_u = []
forecast_confirmed_yhat_l = []

for index, row in forecast_confirmed.iterrows():

    forecast_confirmed_yhat.append(row['yhat'])
    forecast_confirmed_yhat_l.append(row['yhat_lower'])
    forecast_confirmed_yhat_u.append(row['yhat_upper'])


plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(forecast_confirmed_yhat, label='Prediction', color='blue')
plt.plot(forecast_confirmed_yhat_l, label='Prediction lower', color='red')
plt.plot(forecast_confirmed_yhat_u, label='Predicition upper', color='green')
plt.title("Forecast of confirmed cases ")
plt.legend()
plt.show()

#
model_death = Prophet(interval_width=0.99)
model_death.fit(date_death_prophet)
future_death = model_death.make_future_dataframe(periods=30)
forecast_death = model_death.predict(future_death)

dates_forecast_death = []
forecast_death_yhat = []
forecast_death_yhat_u = []
forecast_death_yhat_l = []

for index, row in forecast_death.iterrows():
    dates_forecast_death.append(row['ds'])
    forecast_death_yhat.append(row['yhat'])
    forecast_death_yhat_l.append(row['yhat_lower'])
    forecast_death_yhat_u.append(row['yhat_upper'])


plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(forecast_death_yhat, label='Prediction', color='blue')
plt.plot(forecast_death_yhat_l, label='Prediction lower', color='red')
plt.plot(forecast_death_yhat_u, label='Predicition upper', color='green')
plt.title("Forecast of death cases ")
plt.legend()
plt.show()

model_recovered = Prophet(interval_width=0.99)
model_recovered.fit(date_recovered_prophet)
future_recovered = model_recovered.make_future_dataframe(periods=30)
forecast_recovered = model_recovered.predict(future_recovered)

dates_forecast_recovered = []
forecast_recovered_yhat = []
forecast_recovered_yhat_u = []
forecast_recovered_yhat_l = []

for index, row in forecast_recovered.iterrows():
    dates_forecast_recovered.append(row['ds'])
    forecast_recovered_yhat.append(row['yhat'])
    forecast_recovered_yhat_l.append(row['yhat_lower'])
    forecast_recovered_yhat_u.append(row['yhat_upper'])


plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(forecast_recovered_yhat, label='Prediction', color='blue')
plt.plot(forecast_recovered_yhat_l, label='Prediction lower', color='red')
plt.plot(forecast_recovered_yhat_u, label='Predicition upper', color='green')
plt.title("Forecast of recovered cases")
plt.legend()
plt.show()


plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(forecast_confirmed_yhat, label='Confirmed', color='blue')
plt.plot(forecast_death_yhat, label='Death', color='red')
plt.plot(forecast_recovered_yhat, label='Recovered', color='green')
plt.title("Forecast of Coronavirus cases")
plt.legend()
plt.show()

date_confirmed_prophet_y = []
date_recovered_prophet_y = []
date_death_prophet_y = []

for index, row in date_confirmed_prophet.iterrows():
    date_confirmed_prophet_y.append(row['y'])
for index, row in date_death_prophet.iterrows():
    date_death_prophet_y.append(row['y'])
for index, row in date_recovered_prophet.iterrows():
    date_recovered_prophet_y.append(row['y'])


plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(forecast_confirmed_yhat, label = 'Confirmed forecast')
plt.plot(date_confirmed_prophet_y, label = 'Confirmed')
plt.title("Confirmed vs Confirmed forecast Coronavirus")
plt.legend()
plt.show()

plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(forecast_death_yhat, label = 'Death forecast')
plt.plot(date_death_prophet_y, label = 'Death')
plt.title("Death vs Death forecast Coronavirus")
plt.legend()
plt.show()

plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(forecast_recovered_yhat, label = 'Recovered forecast')
plt.plot(date_recovered_prophet_y, label = 'Recovered')
plt.title("Recovered vs Recovered forecast Coronavirus")
plt.legend()
plt.show()



from statsmodels.tsa.ar_model import AR

date_confirmed = df_date[['Date', 'Confirmed']]
date_death = df_date[['Date', 'Deaths']]
date_recovered = df_date[['Date', 'Recovered']]
print(date_death)
for index, row in date_confirmed.iterrows():
    if row['Confirmed'] is None:
        row['Confirmed'] = 0.0

for index, row in date_death.iterrows():
    if row['Deaths'] is None:
        row['Deaths'] = 0.0

for index, row in date_recovered.iterrows():
    if row['Recovered'] is None:
        row['Recovered'] = 0.0


model_ar_confirmed = AR(np.asanyarray(date_confirmed['Confirmed']))
model_fit_ar_confirmed = model_ar_confirmed.fit()
predict_ar_confirmed = model_fit_ar_confirmed.predict(10, len(date_confirmed) + 40)
print(predict_ar_confirmed)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_confirmed['Confirmed'], label='Confirmed', color='blue')
plt.plot(predict_ar_confirmed, label='Predicted unknown data', color='orange')
plt.plot(predict_ar_confirmed[:len(predict_ar_confirmed)-30], label='Predicted known data', color='red')
plt.title('Confirmed cases vs Predicted Confirmed cases')
plt.legend()
plt.show()

model_ar_death = AR(np.asanyarray(date_death['Deaths']))
model_fit_ar_death = model_ar_death.fit()
predict_ar_death = model_fit_ar_death.predict(10, len(date_death) + 40)
print(predict_ar_death)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_death['Deaths'], label='Death', color='blue')
plt.plot(predict_ar_death, label='Predicted unknown data', color='orange')
plt.plot(predict_ar_death[:len(predict_ar_death)-30], label='Predicted known data', color='red')
plt.title('Death cases vs Predicted Death cases')
plt.legend()
plt.show()

model_ar_recovered = AR(np.asanyarray(date_recovered['Recovered']))
model_fit_ar_recovered = model_ar_recovered.fit()
predict_ar_recovered = model_fit_ar_recovered.predict(10, len(date_recovered) + 40)
print(predict_ar_recovered)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_recovered['Recovered'], label='Recovered', color='blue')
plt.plot(predict_ar_recovered, label='Predicted unknown data', color='orange')
plt.plot(predict_ar_recovered[:len(predict_ar_recovered)-30], label='Predicted known data', color='red')
plt.title('Recovered cases vs Predicted Recovered cases')
plt.legend()
plt.show()

plt.subplot(121)
plt.title("Coronavirus data")
plt.plot(date_confirmed['Confirmed'], label='Confirmed', color='blue')
plt.plot(date_death['Deaths'], label='Deaths', color='red')
plt.plot(date_recovered['Recovered'], label='Recovered', color='green')
plt.legend()
plt.subplot(122)
plt.title("Coronavirus data predicted")
plt.plot(predict_ar_confirmed, color='orange')
plt.plot(predict_ar_confirmed[:len(predict_ar_confirmed)-30], label='Predicted Confirmed ', color='blue')
plt.plot(predict_ar_death, color = 'orange')
plt.plot(predict_ar_death[:len(predict_ar_death)-30], label='Predicted Death', color = 'red')
plt.plot(predict_ar_recovered,  color = 'orange')
plt.plot(predict_ar_recovered[:len(predict_ar_recovered)-30], label='Predicted Recovered', color = 'green')
plt.legend()
plt.show()





import statsmodels.api as sm

fig = plt.figure()
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(date_confirmed['Confirmed'], lags=10, ax=ax1) #
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(date_confirmed['Confirmed'], lags=10, ax=ax2)#
plt.show()

from statsmodels.tsa.arima_model import ARIMA
model_ma_confirmed = ARIMA(np.asanyarray(date_confirmed['Confirmed']),  order=(2,0,0))
model_fit_ma_confirmed = model_ma_confirmed.fit(disp=False)
predict_ma_confirmed = model_fit_ma_confirmed.predict(1, len(date_confirmed)+31)
print(predict_ma_confirmed)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_confirmed['Confirmed'], label='Confirmed', color='blue')
plt.plot(predict_ma_confirmed, label='Predicted unknown data', color='orange')
plt.plot(predict_ma_confirmed[:len(predict_ma_confirmed)-31], label='Predicted known data', color='red')
plt.title('Confirmed cases vs Predicted Confirmed cases')
plt.legend()
plt.show()

fig = plt.figure()

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(date_death['Deaths'], lags=10, ax=ax1) #
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(date_death['Deaths'], lags=10, ax=ax2)#
plt.show()


model_ma_death = ARIMA(np.asanyarray(date_death['Deaths']),  order=(1, 0, 0))
model_fit_ma_death = model_ma_death.fit(disp=False)
predict_ma_death = model_fit_ma_death.predict(1, len(date_death) + 31)
print(predict_ma_death)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_death['Deaths'], label='Death', color='blue')
plt.plot(predict_ma_death, label='Predicted unknown data', color='orange')
plt.plot(predict_ma_death[:len(predict_ma_death)-31], label='Predicted known data', color='red')
plt.title('Death cases vs Predicted Death cases')
plt.legend()
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(date_recovered['Recovered'], lags=10, ax=ax1) #
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(date_recovered['Recovered'], lags=10, ax=ax2)#
plt.show()

model_ma_recovered = ARIMA(np.asanyarray(date_recovered['Recovered']),  order=(2,0, 0))
model_fit_ma_recovered = model_ma_recovered.fit(disp=False)
predict_ma_recovered = model_fit_ma_recovered.predict(1, len(date_recovered) + 31)
print(predict_ma_recovered)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_recovered['Recovered'], label='Recovered', color='blue')
plt.plot(predict_ma_recovered, label='Predicted unknown data', color='orange')
plt.plot(predict_ma_recovered[:len(predict_ma_recovered)-31], label='Predicted known data', color='red')
plt.title('Recovered cases vs Predicted Recovered cases')
plt.legend()
plt.show()


plt.subplot(121)
plt.title("Coronavirus data")
plt.plot(date_confirmed['Confirmed'], label='Confirmed', color='blue')
plt.plot(date_death['Deaths'], label='Deaths', color='red')
plt.plot(date_recovered['Recovered'], label='Recovered', color='green')
plt.legend()
plt.subplot(122)
plt.title("Coronavirus data predicted")
plt.plot(predict_ma_confirmed, color='orange')
plt.plot(predict_ma_confirmed[:len(predict_ma_confirmed)-31], label='Predicted Confirmed ', color='blue')
plt.plot(predict_ma_death, color = 'orange')
plt.plot(predict_ma_death[:len(predict_ma_death)-31], label='Predicted Death', color = 'red')
plt.plot(predict_ma_recovered,  color = 'orange')
plt.plot(predict_ma_recovered[:len(predict_ma_recovered)-31], label='Predicted Recovered', color = 'green')
plt.legend()
plt.show()




from statsmodels.tsa.statespace.sarimax import SARIMAX


model_sarima_confirmed = SARIMAX(np.asanyarray(date_confirmed['Confirmed']),  order=(2,1,0), seasonal_order=(1,1,0,12))
model_fit_sarima_confirmed = model_sarima_confirmed.fit(disp=False, enforce_stationarity=False)
predict_sarima_confirmed = model_fit_sarima_confirmed.predict(1, len(date_confirmed)+31)
print(predict_sarima_confirmed)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_confirmed['Confirmed'], label='Confirmed', color='blue')
plt.plot(predict_sarima_confirmed, label='Predicted unknown data', color='orange')
plt.plot(predict_sarima_confirmed[:len(predict_sarima_confirmed)-31], label='Predicted known data', color='red')
plt.title('Confirmed cases vs Predicted Confirmed cases')
plt.legend()
plt.show()

model_sarima_death = SARIMAX(np.asanyarray(date_death['Deaths']),  order=(1,1,0), seasonal_order=(1,1,0,12))
model_fit_sarima_death = model_sarima_death.fit(disp=False, enforce_stationarity=False)
predict_sarima_death = model_fit_sarima_death.predict(1, len(date_death)+31)
print(predict_sarima_death)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_death['Deaths'], label='Death', color='blue')
plt.plot(predict_sarima_death, label='Predicted unknown data', color='orange')
plt.plot(predict_sarima_death[:len(predict_sarima_death)-31], label='Predicted known data', color='red')
plt.title('Death cases vs Predicted Death cases')
plt.legend()
plt.show()

model_sarima_recovered = SARIMAX(np.asanyarray(date_recovered['Recovered']),  order=(2,1,0), seasonal_order=(1,1,0,12))
model_fit_sarima_recovered = model_sarima_recovered.fit(disp=False, enforce_stationarity=False)
predict_sarima_recovered = model_fit_sarima_recovered.predict(1, len(date_recovered)+31)
print(predict_sarima_recovered)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_recovered['Recovered'], label='Recovered', color='blue')
plt.plot(predict_sarima_recovered, label='Predicted unknown data', color='orange')
plt.plot(predict_sarima_recovered[:len(predict_sarima_recovered)-31], label='Predicted known data', color='red')
plt.title('Recovered cases vs Predicted Recovered cases')
plt.legend()
plt.show()

plt.subplot(121)
plt.title("Coronavirus data")
plt.plot(date_confirmed['Confirmed'], label='Confirmed', color='blue')
plt.plot(date_death['Deaths'], label='Deaths', color='red')
plt.plot(date_recovered['Recovered'], label='Recovered', color='green')
plt.legend()
plt.subplot(122)
plt.title("Coronavirus data predicted")
plt.plot(predict_sarima_confirmed, color='orange')
plt.plot(predict_sarima_confirmed[:len(predict_sarima_confirmed)-31], label='Predicted Confirmed ', color='blue')
plt.plot(predict_sarima_death, color = 'orange')
plt.plot(predict_sarima_death[:len(predict_sarima_death)-31], label='Predicted Death', color = 'red')
plt.plot(predict_sarima_recovered,  color = 'orange')
plt.plot(predict_sarima_recovered[:len(predict_sarima_recovered)-31], label='Predicted Recovered', color = 'green')
plt.legend()
plt.show()

import scipy.stats as stats

#CONFIRMED
spearman_ar_confirmed = stats.spearmanr(date_confirmed['Confirmed'], predict_ar_confirmed[:len(predict_ar_confirmed)-31])[0]
spearman_arima_confirmed = stats.spearmanr(date_confirmed['Confirmed'], predict_ma_confirmed[:len(predict_ma_confirmed)-31])[0]
spearman_sarima_confirmed = stats.spearmanr(date_confirmed['Confirmed'], predict_sarima_confirmed[:len(predict_sarima_confirmed)-31])[0]
print()
print("SPEARMAN CONFIRMED AR: ", spearman_ar_confirmed)
print("SPEARMAN CONFIRMED ARIMA: ", spearman_arima_confirmed)
print("SPEARMAN CONFIRMED SARIMA: ", spearman_sarima_confirmed)

#DEATH
spearman_ar_death = stats.spearmanr(date_death['Deaths'], predict_ar_death[:len(predict_ar_death)-31])[0]
spearman_arima_death = stats.spearmanr(date_death['Deaths'], predict_ma_death[:len(predict_ma_death)-31])[0]
spearman_sarima_death = stats.spearmanr(date_death['Deaths'], predict_sarima_death[:len(predict_sarima_death)-31])[0]
print()
print("SPEARMAN DEATH AR: ", spearman_ar_death)
print("SPEARMAN DEATH ARIMA: ", spearman_arima_death)
print("SPEARMAN DEATH SARIMA: ", spearman_sarima_death)
#RECOVERED

spearman_ar_recovered = stats.spearmanr(date_recovered['Recovered'], predict_ar_recovered[:len(predict_ar_recovered)-31])[0]
spearman_arima_recovered = stats.spearmanr(date_recovered['Recovered'], predict_ma_recovered[:len(predict_ma_recovered)-31])[0]
spearman_sarima_recovered = stats.spearmanr(date_recovered['Recovered'], predict_sarima_recovered[:len(predict_sarima_recovered)-31])[0]
print()
print("SPEARMAN RECOVERED AR: ", spearman_ar_recovered)
print("SPEARMAN RECOVERED ARIMA: ", spearman_arima_recovered)
print("SPEARMAN RECOVERED SARIMA: ", spearman_sarima_recovered)