# CSE 150A Milestone 2

## [Dataset link](https://archive.ics.uci.edu/dataset/312/dow+jones+index)

## [Jupyter Notebook](https://colab.research.google.com/drive/1ysVW8dKDKrU8gJPSCyoAYanBV9dPKj4Y#scrollTo=nFW2Pt4iCL3a)

## Group Members 
Dat To Ung\
Hoang Le\
Uyen Le\
Yifan Zhu\
Dylan Nguyen

## Dataset: 
What is our agent doing in terms of PEAS?\
P (performance measure): How well does our model predict next week's stock price change?\
E (environment): The market and stock data.\
A (actuators): Data processing, cleaning, and creating new attribute variables with the provided data.\
S (sensors): Using the stock market data from ucimlrepo and open/close prices, volume, percent changes\
\
Our agent is a Bayesian Network-based decision-making system that predicts stock price movements using probabilistic reasoning. \
It takes in various market indicators and historical price data as inputs. \
The Bayesian Network models dependencies between these variables, forming conditional probabilities to estimate the likelihood of different price trends. \
\
## Conclusion:
The prediction accuracy of our Naive Bayes model is 41.333%, demonstrating that the model is an improvement over guessing stock trends at random, but not by very much.\ 
Perhaps we would have to find an alternative method to clean the data instead of using placeholder values.
