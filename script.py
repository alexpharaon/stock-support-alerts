# IMPORTING PACKAGES
import os
import numpy as np
import pandas as pd
import datetime as dt
from twilio.rest import Client # to send SMS
import schedule # to schedule the task at a certain time
import time # to set the task to a certain time
from datetime import date
import yfinance as yf
from webserver import keep_alive
from sklearn.cluster import DBSCAN # for building a clustering model
from sklearn import metrics # for calculating Silhouette score
from sklearn.linear_model import LinearRegression
from zigzag import *

# Keep this repl running forever using uptime robot
keep_alive()

# ALGORITHM
tickers = pd.read_csv('support_tickers.csv')
tickers = list(tickers.iloc[:,1].values)
max_acceptable_deviation = 0.05

def get_data(tickers, period: str):
  stock_data = yf.download(tickers=tickers, period=period, interval='1d')
  stock_data = stock_data.reset_index()
  stock_data = stock_data.drop('Date', axis=1)

  return stock_data

def get_zigzags(close: pd.Series(dtype='float64')):
  pivots = peak_valley_pivots(close.values, 0.2, -0.2) 
  ts_pivots = pd.Series(close.values, index=close.index)
  
  ts_pivots = ts_pivots[pivots != 0]
  ts_pivots = ts_pivots[1:] # removes first point in timeseries arbitrarily being assigned as a peak
  ts_pivots = ts_pivots[:-1] # removes first point in timeseries arbitrarily being assigned as a stationary point

  # Seperate peaks and troughs
  peaks = {} # peak_value: peak_index
  troughs = {}

  for i in range(len(ts_pivots.values)):
    if (i % 2) == 0: # even; ie. trough (since first peak removed above)
      troughs[ts_pivots.values[i]] = ts_pivots.index[i]
    else:
      peaks[ts_pivots.values[i]] = ts_pivots.index[i]
  print(troughs)
  return peaks, troughs

def get_clusters(twoD_array: np.array, percentage_radius: float):
  # Set the model and its parameters
  avg_price = float(sum(twoD_array) / len(twoD_array))
  model = DBSCAN(eps=percentage_radius*avg_price, min_samples=3)
  # Fit the model 
  clm = model.fit(twoD_array)
  print(clm.labels_)
  return clm.labels_ # return the cluster labels for each element in the inputed array

def run_regression(clm_labels: np.array, troughs_or_peaks: dict, troughs: bool, ticker: str, deviation, stock_data):
  print('r')
  # Run a linear regression on each cluster
  number_of_clusters = len(set(clm_labels)) # number of clusters
  
  # If there is any '-1' label, reduce the number of clusters by 1 as this is an outlier rather than another cluster set
  for label in clm_labels:
    if label == -1:
      number_of_clusters -= 1 # there is a -1 element so break the loop and reduce the number of clusters by one
      break

  # Loop through each cluster and run a regression on that cluster
  for cluster_i in range(number_of_clusters):
    Y = np.array([]) # trough values
    X = np.array([]) # index of the trough values
    
    # Assign the appropriate data to X and Y
    for i in range(len(clm_labels)):
      if clm_labels[i] == cluster_i:
        Y = np.append(Y,list(troughs_or_peaks.keys())[i])
        X = np.append(X,troughs_or_peaks[list(troughs_or_peaks.keys())[i]])
    
    X = np.array(X).reshape(-1,1)  # values converts it into a numpy array
    Y = np.array(Y).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X,Y)  # perform linear regression
    Y_pred = linear_regressor.predict(np.array(stock_data.index.stop-1).reshape(-1,1))  # make predictions

    print((1+deviation)*Y_pred)
    print(stock_data.iloc[-1].loc['Adj Close'])

    if troughs and (1+deviation)*Y_pred > stock_data.iloc[-1].loc['Adj Close'] > (1-deviation)*Y_pred:
      send_sms(text=f"{ticker} reached a strong support line yesterday. You may want to consider buying today.", recipients=['+4412345678'])

    elif troughs and (1+deviation+0.02)*Y_pred > stock_data.iloc[-1].loc['Adj Close'] >= (1+deviation)*Y_pred:
      send_sms(text=f"{ticker} is approaching a strong support line. You may want to consider buying soon.", recipients=['+4412345678'])

def send_sms(text: str, recipients: list):
  for recipient in recipients:
    account_sid = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' # remember to edit this to insert your details (you can find this on Twilio)
    auth_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'    # remember to edit this to insert your details (you can find this on Twilio)
    client = Client(account_sid, auth_token)

    message = client.messages \
                    .create(
                        body=text,
                        from_='XXXXXXXX',
                        to=recipient
                    ) # remember to edit this to insert the number you want to text from (you can find this on Twilio)

    print(message.sid) # prints a code if the message has been successfully sent 

def run_script():
  for ticker in tickers:

    # Get the data
    stock_data = get_data(ticker,'5y')

    # Skip this ticker if there is no data at all
    if len(stock_data.index) == 0:
      continue

    peaks, troughs = get_zigzags(stock_data.loc[:,'Adj Close'])

    # Try to get clusters. If there aren't any, it will move on to except
    try:
      cluster_labels = get_clusters(np.array(list(troughs.keys())).reshape(-1,1), max_acceptable_deviation)
      if sum(cluster_labels) == - len(cluster_labels):
        # no clusters as everything is deemed an outlier
        continue
        
    except:
      continue # something went wrong so move on to the next ticker

    run_regression(cluster_labels, troughs, True, ticker, 0.03, stock_data)


# SCHEDULE WHEN TO RUN EVERYTHING 
## The schedule does not clear even if you cancel a run and start it again so need to clear the schedule every time it is run and then re-declare the schedule
schedule.clear()

schedule.every().saturday.at("22:00").do(run_script) # Run at 10pm UTC time (ie. 6pm NY time and 2am Dubai Time)
schedule.every().tuesday.at("22:00").do(run_script) # Run at 10pm UTC time (ie. 6pm NY time and 2am Dubai Time)
schedule.every().wednesday.at("22:00").do(run_script) # Run at 10pm UTC time (ie. 6pm NY time and 2am Dubai Time)
schedule.every().thursday.at("22:00").do(run_script) # Run at 10pm UTC time (ie. 6pm NY time and 2am Dubai Time)
schedule.every().friday.at("22:00").do(run_script) # Run at 10pm UTC time (ie. 6pm NY time and 2am Dubai Time)


# Keep the schedule running perpetually. The while loop checks if there is task the scheduler must run.
while True:
    schedule.run_pending()
    time.sleep(1)
