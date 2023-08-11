import sys
import os
  
os.system(f"{sys.executable} -m pip install -U inline")
os.system(f"{sys.executable} -m pip install -U matplotlib")
os.system(f"{sys.executable} -m pip install -U statsmodels")

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
#import statsmodels.formula.api as smf            # statistics and econometrics
#import statsmodels.tsa.api as smt
#import statsmodels.api as sm
#import scipy.stats as scs
#from tqdm import tqdm_notebook
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
#import matplotlib.dates as mdates
import inline
import pytd


TD_API_KEY = "apikey"
apikey = "apikey"

client = pytd.Client(apikey='apikey', endpoint='https://api.treasuredata.com/', database='database', default_engine='presto')

def data():
#    data = client.query("")
    data = client.query("query")
    a = pd.json_normalize(data, "data")
    a.rename(columns={0: "move_in_date", 1: "total_count"}, inplace=True)
    df = pd.DataFrame(a)
    
    if 'move_in_date' not in df.columns:
        print("Column 'move_in_date' not found in the dataframe.")
        print("Available columns:", df.columns)
        return None
    
    df['move_in_date'] = pd.to_datetime(df['move_in_date'])
    #df = df.loc[df['move_in_date'] >= '2020-01-01']
    
    df.sort_values('move_in_date', inplace=True)
    
    return df

def generate_forecast(df):
    df.set_index('move_in_date', inplace=True)

    # Fit ARIMA model
    model = ARIMA(df['total_count'], order=(1, 0, 0))
    model_fit = model.fit()

    # Predict next month's values
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]  # Generate future dates
    forecast = model_fit.predict(start=len(df), end=len(df) + 29)  # Adjust the end value based on the desired forecast length

    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})

    # Print the forecasted values
    print(forecast_df)

# Assuming 'result_df' is the dataframe you want to generate a forecast for


def generate_monthly_forecast(df):
    monthly_data = df.resample('M').sum()

    # Predict next month's number
    last_date = monthly_data.index[-1]
    next_month = last_date + pd.DateOffset(months=1)
    model = ARIMA(monthly_data['total_count'], order=(1, 0, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)

    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame({'Date': [next_month], 'Forecast': forecast})

    # Print the monthly data and the forecasted value for the next month
    print("Monthly Data:")
    print(monthly_data)
    print("\nForecast:")
    print(forecast_df)



def generate_forecast_plot_multi_month(df):
    monthly_data = df.resample('M').sum()

    model = ARIMA(monthly_data['total_count'][:-2], order=(1, 0, 0))
    model_fit = model.fit()

    start_index = len(monthly_data) - 12  # Number of months you want to go back (6 months + current month)
    end_index = len(monthly_data) - 1
    forecast_past = model_fit.predict(start=start_index, end=end_index)

    future_index = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=3, freq='M')
    forecast_future = model_fit.forecast(steps=3)[0]

    forecast_values = pd.concat([forecast_past, pd.Series(forecast_future, index=future_index)])

    plt.figure(figsize=(15, 7))
    plt.plot(monthly_data.index - pd.DateOffset(months=1), monthly_data['total_count'], label='Actual')
    plt.plot(forecast_values.index - pd.DateOffset(months=1), forecast_values, color='r', linestyle='--', label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Move Ins')
    plt.title('Monthly Data and Forecast')
    plt.legend(loc='upper left')
    
    # Customize x-axis tick labels to show every other month
    plt.xticks(rotation=90)
    ax = plt.gca()
    #ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) #Can set this to one depending on what they want axis to be
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.grid(True)
    plt.show()



def generate_actual_vs_predicted(df):
    monthly_data = df.resample('M').sum()
    model = ARIMA(monthly_data['total_count'][:-2], order=(1, 0, 0))
    model_fit = model.fit()

    start_index = len(monthly_data) - 1  # Number of months you want to go back (6 months + current month) Was originally 12, now 1.
    end_index = len(monthly_data) - 1
    forecast_past = model_fit.predict(start=start_index, end=end_index)

    future_index = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=3, freq='M')
    forecast_future = model_fit.forecast(steps=3)[0]

    forecast_values = pd.concat([forecast_past, pd.Series(forecast_future, index=future_index)])

    merged_table = monthly_data.merge(forecast_values.to_frame(), left_index=True, right_index=True)
    final_table = merged_table.rename(columns={0: 'Predicted Values'})

    final_table['Percentage Difference'] = final_table[['total_count', 'Predicted Values']].pct_change(axis=1)['Predicted Values'] * 100
    final_table = final_table.round({'Predicted Values': 2})
    final_table = final_table.round({'Percentage Difference': 2})
    final_table['Version_History'] = 'v1' #change. this to whatever version we want to runit as, since this is the first set, we will keep v1
    final_table['Date'] = final_table.index.strftime('%m-%d-%Y')


    return final_table


def main():
    result_df = data()
    #if result_df is not None:
    #    print(result_df)
    generate_forecast(result_df)
    #generate_monthly_forecast(result_df)
    #generate_forecast_plot_one_month(result_df)
    #generate_forecast_plot_multi_month(result_df)
    forecast_table = generate_actual_vs_predicted(result_df)
    print(forecast_table)
