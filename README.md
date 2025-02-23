
# PROJECT CSE 150A: AI PROBABILISTIC MODELS

## [Dataset link](https://archive.ics.uci.edu/dataset/312/dow+jones+index)

## [Jupyter Notebook](https://colab.research.google.com/drive/1ysVW8dKDKrU8gJPSCyoAYanBV9dPKj4Y#scrollTo=nFW2Pt4iCL3a)

## Group Members 
Dat To Ung\
Hoang Le\
Uyen Le\
Yifan Zhu\
Dylan Nguyen

---

## Project Abstraction: 
In this project, we propose a utility-based AI agent that predicts market trends using a time-series dataset from UCI containing daily price data that was published on October 24th, 2014. Instead of performing a continuous-value regression on opening/closing prices or trading volume, we quantize these features into discrete intervals, by defining thresholds for small, medium, large price changes. This allows our agent to classify each day’s market behavior as “Up/Down/Stable” capturing directional trends rather than generating a single numeric forecast. Operating within the PEAS framework, the environment is the stock market, the performance measure is risk-adjusted profit, the actuators are buy/sell/hold trades, and the sensors are historical market observations. The agent continually updates its belief over market states and selects the action that maximizes expected returns under uncertainty. Through this probabilistic approach, we aim to demonstrate how interpreting and exploiting hidden market regimes can lead to more informed and adaptive trading strategies than a simple static predictor. In a regression task, we would just simply predict a numerical value such as the price of the stock or the percentage change in the price. But our task is different from a regression task in that our models will predict where the stock price will fall today. We want our model to detect the trends rather than the specific numbers. 

---

## PEAS Framework  

Our AI agent operates within the **PEAS framework** (Performance measure, Environment, Actuators, Sensors) as follows:  

### **Performance Measure:**  
The agent’s goal is to **predict market trends** to make optimal **buy/sell/hold** decisions rather than predicting explicit numeric forecasts. Success is measured by **risk-adjusted profit** based on:  
- **Prediction accuracy** of classifying each day's market behavior as **Up/Down/Stable**.  
- **Correct identification** of **bullish or bearish market regimes** over a given time frame.  
- **Error in estimating the percentage change in price** for the next week (lower error is better).  

### **Environment:**  
The environment is the **stock market**, modeled using the **Dow Jones Index dataset from UCI**, containing daily price and volume data from 2014.  

### **Actuators:**  
The agent interacts with the market by executing **three possible trading actions**:  
- **Buy** – Purchase stocks in anticipation of a price increase.  
- **Sell** – Sell stocks to lock in profits or prevent losses.  
- **Hold** – Maintain the current position when there is uncertainty.  

### **Sensors:**  
The agent gathers observations from **historical market data**, including **opening and closing prices, high and low prices, and trading volume**. These features are **quantized into discrete intervals** (e.g., small, medium, large price changes) to categorize daily market behavior as **Up, Down, or Stable**. The agent **continuously updates its belief over market states** based on these observations.  

---

## What is the “World” Like?  
The agent operates in a **financial market** where stock prices fluctuate due to complex, often hidden factors. The world is **uncertain and dynamic**, meaning that the same market conditions may not always lead to identical outcomes. The agent must **learn from past patterns** to infer hidden market trends. By **interpreting market shifts probabilistically**, the agent aims to make informed trades that yield **higher cumulative returns** over time.  

---

## **Agent Type: Utility-Based AI Agent**  

Our AI agent is a **utility-based agent** because it selects actions (**Buy/Sell/Hold**) based on **maximizing expected returns under uncertainty**. By predicting stock movements (**Up/Down/Stable**), it evaluates different actions using a **utility function** that considers **profitability, risk, and market conditions**.  

Unlike a **goal-based agent** that focuses on achieving a fixed objective, our agent **compares the potential outcomes of different decisions** and chooses the one with the **highest expected reward**. This approach allows for **more adaptive and informed trading strategies** rather than relying on static predictions.  

---

## **Agent Setup and Probabilistic Modeling**  

Our agent is designed as a **probabilistic classifier** that predicts **market trends** by analyzing **historical stock price movements**. Instead of using **continuous-value regression**, we **discretize price changes** into categories (**Up, Down, Stable**) and model the **probability distribution** of these outcomes.  

### **Agent Setup:**  

1. **Feature Engineering:**  
   - We extract key **stock price features**: **opening price, closing price, highest price, lowest price, and trading volume**.  
   - These features are **quantized into discrete intervals** representing **small, medium, or large changes** in price.  

2. **Probabilistic Modeling Approach:**  
   - The agent uses **probabilistic inference** to determine the **likelihood of market trends** given the **volume change and price change**.  
   - It estimates **P(Trend | Market Features)**, where the **Trend** is **Up/Down/Stable**.  
   - This allows the model to **infer hidden market regimes** and adjust predictions dynamically.  

3. **Decision-Making:**  
   - The agent selects **Buy/Sell/Hold** actions based on its **estimated probabilities**.  
   - If **P(Up) is high**, the agent may **Buy**; if **P(Down) is high**, it may **Sell**; if **P(Stable)**, it may **Hold**.  
   - This decision process **relies on probability distributions** rather than **deterministic rules**.  

### **How It Fits in Probabilistic Modeling:**  
Our agent applies **probabilistic classification** rather than **rule-based heuristics**. By **estimating the probability** of stock movements, it accounts for **uncertainty in financial markets** and avoids **overfitting to specific trends**. This makes it **more adaptable** compared to simple deterministic models.  

---

## Methods Overview:
### Data Exploration

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

### Data Preprocessing
1.  Imputation:

For data preprocessing, there are some missing values for 'percent_change_volume_over_last_wk' and 'previous_weeks_volume':

![image](https://github.com/user-attachments/assets/c8058998-71ee-4295-86c7-6cf8271824ca)

So we decided to fill in the missing values with the median of each type of stock. 

2. Standardization
 noticed that there are a lot of '$' signs in the data, so we decided to remove all the dollar signs ('$'). Finally, we added 3 more new columns which are 'PriceChange',	'VolumeChange', and 'MarketTrend'. 

3. Feature expansion:
The data for each of these columns is from converting the continuous values of 'percent_change_price', 'percent_change_next_weeks_price', and 'percent_change_volume_over_last_wk' to discrete values.  

### Finalized dataframe:
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
**Overview**

This model uses the dataset to compute the probability of each kind of Market Trend of each stock given the Volume Change (in categorical) and the Price Change (in categorical). Formula:

$$
P(\text{MarketTrend}=y \mid \text{VolumeChange}=x_1, \text{PriceChange}=x_2) = \frac{\text{Number of } (\text{MarketTrend}=y, \text{VolumeChange}=x_1, \text{PriceChange}=x_2)}{\text{Number of } (\text{VolumeChange}=x_1, \text{PriceChange}=x_2)}
$$

Then, when the user gives the Volume and Price Change (in categorical) and asks the agent to suggest selling/buying/holding stock, the model will try to find the market trend of this stock based on the probability and give the user a suggestion.

**Code**
```
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
```
# Results:
## Model 1: Bayesian Network

Currently, our agent only has 41.33% accuracy. We think this accuracy is not too high because we assumed that the price change and the volume change are independent. Also, when we converted the volume change, the price change, and the real market trend from numerical to categorical, we used the threshold is 1.5, which is not the most efficient for this model. We think we could improve our model if we have a larger dataset to train and test this model as well as we could try to figure out the most efficient threshold for standardizing. 
# Conclusion:
## Model 1: Bayesian Network
The prediction accuracy of our Naive Bayes model is 41.333%, demonstrating that the model is an improvement over guessing stock trends at random, but not by very much. Perhaps we would have to find an alternative method to clean the data instead of using placeholder values.  We can also improve our agent with a bigger dataset and more features. Our agent can have more historical stock data with more data in our dataset. With the improvement of the historical stock data, the agent can calculate the probability more accurately and more reliably. Improving our features would also make our agent more reliable because our current agent is oversimplifying the market dynamic by assuming the price change and volume change are independent. By having more features, we can improve our agent’s structure which also boosts the accuracy of our agent. There are many challenges that our agent can have, one of them is the independence assumption. Since this is our first agent, we are assuming the volume change and the price change are independent. But in the real world, these features are correlated, and ignoring the relationship can lead to the underperformance of our agent. With the limitation of our dataset, our agent will be overfitting and have bad predictions, which can lead to poor decision-making. Also, using a fixed threshold is not optimal for all types of stocks, which can also lead to bad output for our agent.
