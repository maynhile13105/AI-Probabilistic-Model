# CSE 150A Milestone 2

## [Dataset link](https://archive.ics.uci.edu/dataset/312/dow+jones+index)

## [Jupyter Notebook](https://colab.research.google.com/drive/1ysVW8dKDKrU8gJPSCyoAYanBV9dPKj4Y#scrollTo=nFW2Pt4iCL3a)

## Group Members 
Dat To Ung\
Hoang Le\
Uyen Le\
Yifan Zhu\
Dylan Nguyen

# Project Abstraction: 
In this project, we propose a utility-based AI agent that predicts market trends using a time-series dataset from UCI containing daily price data that was published on October 24th, 2014. Instead of performing a continuous-value regression on opening/closing prices or trading volume, we quantize these features into discrete intervals, by defining thresholds for small, medium, large price changes. This allows our agent to classify each day’s market behavior as “Up/Down/Stable” capturing directional trends rather than generating a single numeric forecast. Operating within the PEAS framework, the environment is the stock market, the performance measure is risk-adjusted profit, the actuators are buy/sell/hold trades, and the sensors are historical market observations. The agent continually updates its belief over market states and selects the action that maximizes expected returns under uncertainty. Through this probabilistic approach, we aim to demonstrate how interpreting and exploiting hidden market regimes can lead to more informed and adaptive trading strategies than a simple static predictor. In a regression task, we would just simply predict a numerical value such as the price of the stock or the percentage change in the price. But our task is different from a regression task in that our models will predict where the stock price will fall today. We want our model to detect the trends rather than the specific numbers. 

# Methods Overview:
## Data Exploration
## Data Preprocessing
# Finalized dataframe:
<img width="1236" alt="df" src="https://github.com/user-attachments/assets/1f019254-3e53-4a18-97dc-7753d36507bb" /><img width="436" alt="df-continue" src="https://github.com/user-attachments/assets/470dac6d-0715-4a46-8041-5e59e50bfc71" />

- quarter: the yearly quarter (1 = Jan-Mar; 2 = Apr=Jun).
- stock: the stock symbol (see above)
- date: the last business day of the work (this is typically a Friday)
- open: the price of the stock at the beginning of the week
- high: the highest price of the stock during the week
- low: the lowest price of the stock during the week
- close: the price of the stock at the end of the week
- volume: the number of shares of stock that traded hands in the week
- percent_change_price: the percentage change in price throughout the week
- percent_chagne_volume_over_last_wek: the percentage change in the number of shares of stock that traded hands for this week compared to the previous week
- previous_weeks_volume: the number of shares of stock that traded hands in the previous week
- next_weeks_open: the opening price of the stock in the following week
- next_weeks_close: the closing price of the stock in the following week
- percent_change_next_weeks_price: the percentage change in price of the stock in the following week days_to_next_dividend: the number of days until the next dividend
- percent_return_next_dividend: the percentage of return on the next dividend
- PriceChange
- VolumeChange
- MarketTrend

## Model 1: Bayesian Network
What is our agent doing in terms of PEAS?\
P (performance measure): How well does our model predict next week's stock price change?\
E (environment): The market and stock data.\
A (actuators): Data processing, cleaning, and creating new attribute variables with the provided data.\
S (sensors): Using the stock market data from ucimlrepo and open/close prices, volume, percent changes\
\
Our agent is a Bayesian Network-based decision-making system that predicts stock price movements using probabilistic reasoning. \
It takes in various market indicators and historical price data as inputs. \
The Bayesian Network models dependencies between these variables, forming conditional probabilities to estimate the likelihood of different price trends. 
# Results:
## Model 1: Bayesian Network
# Conclusion:
## Model 1: Bayesian Network
The prediction accuracy of our Naive Bayes model is 41.333%, demonstrating that the model is an improvement over guessing stock trends at random, but not by very much. Perhaps we would have to find an alternative method to clean the data instead of using placeholder values.
