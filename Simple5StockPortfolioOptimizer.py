from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime

assets = ['AAPL', 'AMZN', 'GOOG', 'NFLX', 'META']

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

stock_start_date = '2013-01-01'
today = datetime.today().strftime('%Y-%m-%d')

df = pd.DataFrame()

for stock in assets:
    df[stock] = web.DataReader(stock, data_source='yahoo', start = stock_start_date, end = today)['Adj Close']

title = 'Portfolio Adj Close Price History'

my_stocks = df

returns = df.pct_change()

cov_matrix_annual = returns.cov() * 252
port_varience = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_volatility = np.sqrt(port_varience)

portfolio_simple_annual_return = np.sum(returns.mean() * weights) * 252

percent_var = str(round(port_varience, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolio_simple_annual_return, 2) * 100) + '%'

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)

allocation, leftover = da.lp_portfolio()

print('Discrete Allocation:', allocation)
print('Funds Remaining: ${:.2f}'.format(leftover))



