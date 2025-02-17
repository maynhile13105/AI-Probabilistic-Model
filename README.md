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

There are 30 types of stocks in our dataset with 750 rows. 

![image](https://github.com/user-attachments/assets/00ac8659-f2ac-4db8-b5d9-f27a19fa87ba)


The below image is the descriptive of our dataset.
![image](https://github.com/user-attachments/assets/de0f54e3-0ae0-449f-9ec5-3dc6b88f2275)

The below plot shows the closing prices of multiple stocks over time, demonstrating stable trends with moderate fluctuations. Some stocks have consistently higher values, while other stocks remain in lower price ranges.
![closepriceovertime](images/closepriceovertime.png)
\
The plot illustrates trading volume variations over time, showing significant fluctuations for certain stocks. As you can see, some stocks experience periodic spikes.
![trading_over_time](images/trading_over_time.png)
\
The following heatmap visualizes the correlation between different stock-related features; red signifies positive correlation, blue signifies negative correlation, and white signifies no correlation.
![correlation_heatmap](images/correlation_heatmap.png)

## Data Preprocessing
For data preprocessing, there are some missing values for 'percent_change_volume_over_last_wk' and 'previous_weeks_volume':
![image](https://github.com/user-attachments/assets/c8058998-71ee-4295-86c7-6cf8271824ca)

So we decided to fill in the missing values with the median of each type of stock. We also noticed that there are a lot of '$' signs in the data, so we decided to remove all the dollar signs ('$'). Finally, we added 3 more new columns which are 'PriceChange',	'VolumeChange', and 
 'MarketTrend'. The data for each of these columns is from converting the continuous values of 'percent_change_price', 'percent_change_next_weeks_price', and 'percent_change_volume_over_last_wk' to discrete values.  


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
- percent_change_next_weeks_price: the percentage change in price of the stock in the following week
- days_to_next_dividend: the number of days until the next dividend
- percent_return_next_dividend: the percentage of return on the next dividend
- PriceChange: categorizes percent_change_price as "Down" if it's below -1.5, "Up" if above 1.5, and "Stable" otherwise.
- VolumeChange: follows the same logic but based on percent_change_volume_over_last_wk.
- MarketTrend: determined using percent_change_next_weeks_price, labeling it as "Bearish" if below -1.5, "Bullish" if above 1.5, and "Neutral" otherwise.

## Model 1: Bayesian Network
**What is our agent doing in terms of PEAS?**\
P (performance measure): How well does our model predict next week's stock price change?\
E (environment): The market and stock data.\
A (actuators): Data processing, cleaning, and creating new attribute variables with the provided data.\
S (sensors): Using the stock market data from ucimlrepo and open/close prices, volume, percent changes\
**What is the “world” like?**\
The agent operates in a financial market where stock prices fluctuate due to complex, often hidden factors. The world is uncertain and dynamic, meaning that the same market conditions may not always lead to identical outcomes. The agent must learn from past patterns to infer hidden market trends. y interpreting market shifts probabilistically, the agent aims to make informed trades that yield higher cumulative returns over time.\
\
Our agent is a Bayesian Network-based decision-making system that predicts stock price movements using probabilistic reasoning. \
It takes in various market indicators and historical price data as inputs. \
The Bayesian Network models dependencies between these variables, forming conditional probabilities to estimate the likelihood of different price trends. 

# Code for model 1: 
```
import random
class BayesianNetwork:
    def __init__(self, nodes):
        self.nodes = nodes  # List of nodes (variables)
        self.parents = {node: [] for node in nodes}  # Parent relationships
        self.cpt = {}  # Store CPT per stock

    def add_edge(self, parent, child):
        """Define dependency relationships between nodes."""
        self.parents[child].append(parent)

    def set_cpt(self, stock, cpt):
        """Set the conditional probability table, specific to a stock."""
        if stock not in self.cpt:
            self.cpt[stock] = {}
        self.cpt[stock] = cpt

    def get_probability(self, stock, evidence):
        """Compute the probability of a node given the evidence and stock-specific CPT."""
        if stock not in self.cpt:
            return 1/3  # Return 1/3 probability if stock CPT is missing

        # Compute conditional probability using Bayes' Theorem
        key = tuple(evidence.values())
        probabilities = {}
        # If evidence is not in CPT, return 1/3 for each key of market trend
        # Else return the {market_trend,P(market_trend|evidences)
        if key not in self.cpt[stock]:
          probabilities = {k: 1/3 for k in ['Low', 'High', 'Medium']}
        else:
          for k, p in self.cpt[stock][key].items():
            probabilities[k] = p
        return probabilities

    def infer(self, stock, evidence):
      """Perform inference to determine the most likely market trend for a given stock."""
      probabilities = self.get_probability(stock, evidence)

      # Get max probability value
      max_prob = max(probabilities.values())

      # Get all trends that share the max probability
      best_trends = [k for k, v in probabilities.items() if v == max_prob]

      if len(best_trends) == 3:
          return 'Neutral'
      elif len(best_trends) == 2:
        if 'Bearish' in best_trends and 'Bullish' in best_trends:
          return random.choice(best_trends)
        elif 'Bearish' in best_trends and 'Neutral' in best_trends:
          return 'Bearish'
        else:
          return 'Neutral'
      else:
          return best_trends[0]  # Return the single best trend

    def suggested_decision(self, MarketTrend_pred):
      if MarketTrend_pred == 'Bearish':
        return 'Sell'
      elif MarketTrend_pred == 'Bullish':
        return 'Buy'
      else:
        return 'Hold'

# Compute new CPTs per stock
cpt_market_trend_per_stock = {}

# Group data by stock and compute probabilities
grouped = train_df.groupby("stock")

for stock, stock_df in grouped:
    # Compute joint frequency counts
    joint_counts = stock_df.groupby(['VolumeChange', 'PriceChange', 'MarketTrend']).size()

    # Compute conditional probabilities P(MarketTrend | VolumeChange, PriceChange)
    cpt_stock = joint_counts.div(joint_counts.groupby(level=[0, 1]).transform('sum')).unstack().fillna(0).to_dict()
    print(cpt_stock)
    # Store CPT for this stock
    cpt_market_trend_per_stock[stock] = cpt_stock

# Display newly computed CPTs per stock
cpt_market_trend_per_stock


# Compute new CPTs per stock
cpt_market_trend_per_stock = {}

# Group data by stock and compute probabilities
grouped = train_df.groupby("stock")

for stock, stock_df in grouped:
    # Compute joint frequency counts
    joint_counts = stock_df.groupby(['VolumeChange', 'PriceChange', 'MarketTrend']).size()

    # Compute conditional probabilities P(MarketTrend | VolumeChange, PriceChange)
    cpt_stock = joint_counts.div(joint_counts.groupby(level=[0, 1]).transform('sum')).unstack().fillna(0).to_dict()
    print(cpt_stock)
    # Store CPT for this stock
    cpt_market_trend_per_stock[stock] = cpt_stock

# Display newly computed CPTs per stock
cpt_market_trend_per_stock
```
# Results:
## Model 1: Bayesian Network

Currently, our agent only has 41.33% accuracy, which is not too high. One reason is that the dataset is not too big; it only has 750 rows of data and we also perform on a Naive Bayes model. There are still a lot of things to improve the accuracy of our agent. 
# Conclusion:
## Model 1: Bayesian Network
The prediction accuracy of our Naive Bayes model is 41.333%, demonstrating that the model is an improvement over guessing stock trends at random, but not by very much. Perhaps we would have to find an alternative method to clean the data instead of using placeholder values.
We can also improve our agent with a bigger dataset and more features. 
