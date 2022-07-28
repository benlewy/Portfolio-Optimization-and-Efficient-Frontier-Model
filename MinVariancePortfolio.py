import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
plt.style.use('seaborn-colorblind')

# user input
list_of_tickers = []
input_ticker1 = input('Enter the first ticker in portfolio ').upper()
input_ticker2 = input('Enter the second ticker in portfolio ').upper()
input_ticker3 = input('Enter the third ticker in portfolio ').upper()
num_sims = int(input('How many simulated portfolios would you like? '))

list_of_tickers.append(input_ticker1)

tickers = [input_ticker1, input_ticker2, input_ticker3]

# extracting the data
df = yf.download(tickers, start="2000-12-01", end=date.today())

risk_free_rate = 0.0278

df = np.log(1 + df['Adj Close'].pct_change())


# finds portfolio returns with random weight
def portfolio_returns(weights):
    return (np.dot(df.mean(), weights)) * 252


# finds portfolio standard deviation with random weight
def portfolio_stdev(weights):
    return (np.dot(np.dot(df.cov(), weights), weights)) ** (1 / 2) * np.sqrt(252)


# creates random weights to assign to simulated portfolios
def weights_creator(df):
    rand = np.random.random(len(df.columns))
    rand /= rand.sum()
    return rand


# creates portfolio based on weights
returns = []
stdev = []
w = []
sharpe = []

for i in range(num_sims):
    weights = weights_creator(df)
    returns.append(portfolio_returns(weights))
    stdev.append(portfolio_stdev(weights))
    w.append(weights)
    sharpe.append((returns[i] - (risk_free_rate/252) / stdev[i]))

print(sharpe)
index_of_lowest = stdev.index(min(stdev))
stdev_of_lowest = stdev[index_of_lowest]
returns_of_lowest = returns[index_of_lowest]
lwi = w[index_of_lowest]
lowest_weights = list(np.around(np.array(lwi), 2))

high_sharpe = sharpe.index(max(sharpe))
print(high_sharpe)

# plot the efficient frontier
plt.figure(figsize=(9, 8))
ax = plt.gca()
plt.scatter(stdev, returns, c=stdev, cmap='winter_r', marker='o', s=10, alpha=0.3)
plt.scatter(min(stdev), returns[stdev.index(min(stdev))], c='black', label='Minimum Variance Portfolio')
plt.text(.75, .4,
         f'Portfolio Weights:'
         f'\n{tickers[0]}: {lowest_weights[0]*100}%'
         f'\n{tickers[1]}: {lowest_weights[1]*100}%'
         f'\n{tickers[2]}: {lowest_weights[2]*100}%',
         transform=ax.transAxes)
plt.text(.2, 1.08,
         f'Return of lowest risk portfolio: {returns_of_lowest}\n'
         f'StDev of lowest risk portfolio: {stdev_of_lowest}',
         transform=ax.transAxes)
plt.title("Efficient Frontier")
plt.xlabel("Portfolio StDev (Risk)")
plt.ylabel("Portfolio Return")
plt.legend(labelspacing=1.2)
plt.show()

