import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from fbprophet import Prophet


df = pd.read_csv('datasets/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)




df_date = df.groupby(["Date"])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df_country = df.groupby(["Country"])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

print(df_date.tail())
print(df_country.tail())

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


for index, row in df_country.iterrows():

    country_x_ticks.append(row['Country'])
    country_confirmed.append(row['Confirmed'])
    country_deaths.append(row['Deaths'])
    country_recovered.append(row['Recovered'])



# plt.xticks(np.arange(len(date_x_ticks[:10])), date_x_ticks[:10])
# plt.xlabel('Dates')
# plt.ylabel('Cases')
# plt.plot(date_confirmed[:10], label='Confirmed', color='blue')
# plt.plot(date_deaths[:10], label='Deaths', color='red')
# plt.plot(date_recovered[:10], label='Recovered', color='green')
# plt.title("Coronavirus cases in the world by date")
# plt.legend()
# plt.show()



#plt.xticks(np.arange(len(date_x_ticks)), date_x_ticks)
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_confirmed, label='Confirmed', color='blue')
plt.plot(date_deaths, label='Deaths', color='red')
plt.plot(date_recovered, label='Recovered', color='green')
plt.title("Coronavirus cases in the world by date")
plt.legend()
plt.show()

#
# plt.xticks(np.arange(len(country_x_ticks[:5])), country_x_ticks[:5])
# plt.xlabel('Countries')
# plt.ylabel('Cases')
# plt.bar(np.arange(len(country_confirmed[:5])), country_confirmed[:5], align='center', alpha=0.5, color='blue', label='Confirmed')
# plt.bar(np.arange(len(country_deaths[:5])), country_deaths[:5], align='center', alpha=0.5, color='red', label='Deaths')
# plt.bar(np.arange(len(country_recovered[:5])), country_recovered[:5], align='center', alpha=0.5, color='green', label='Recovered')

plt.xlabel('Countries')
plt.ylabel('Cases')
plt.bar(np.arange(len(country_confirmed)), country_confirmed, align='center', alpha=0.5, color='blue', label='Confirmed')
plt.bar(np.arange(len(country_deaths)), country_deaths, align='center', alpha=0.5, color='red', label='Deaths')
plt.bar(np.arange(len(country_recovered)), country_recovered, align='center', alpha=0.5, color='green', label='Recovered')
plt.title("Coronavirus cases in the world by country")
plt.legend()
plt.show()
#
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