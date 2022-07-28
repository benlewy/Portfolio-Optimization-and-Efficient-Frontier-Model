import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
plt.style.use('seaborn-colorblind')

# Create our portfolio of equities
tickers = []
number_of_tickers = int(input("How many stocks in your portfolio? "))

for x in range(number_of_tickers):
    tickers.append(input("What ticker do you want to add to your portfolio? ").upper())

# user can specify the number of portfolios to simulate
num_portfolios = int(input("How many portfolios would you like to simulate? "))
print('\n')
print('Evaluating your portfolio...')

# Download closing prices
data = pdr.get_data_yahoo(tickers, start="2012-01-01", end=dt.date.today())['Close']

# From the closing prices, calculate periodic returns
returns = data.pct_change()


# Define function to calculate returns, volatility
def portfolio_annualized_performance(weights, avg_returns, cov_matrix):
    # Given the avg returns, weights of equities calc. the portfolio return
    returns = np.sum(avg_returns * weights) * 252

    # Standard deviation of portfolio (using dot product against covariance, weights)
    # 252 trading days
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    # Initialize array of shape 3 x N to store our results,
    # where N is the number of portfolios we're going to simulate
    results = np.zeros((3, num_portfolios))

    # Array to store the weights of each equity
    weight_array = []

    for i in range(num_portfolios):
        # Randomly assign floats to our 4 equities
        weights = np.random.random(len(tickers))

        # Convert the randomized floats to percentages (summing to 100)
        weights /= np.sum(weights)

        # Add to our portfolio weight array
        weight_array.append(weights)

        # Pull the standard deviation, returns from our function above using
        # the weights, mean returns generated in this function
        portfolio_std_dev, portfolio_return = portfolio_annualized_performance(weights, mean_returns, cov_matrix)

        # Store output
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return

        # Sharpe ratio
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    return results, weight_array


mean_returns = returns.mean()

cov_matrix = returns.cov()



# Risk-free rate (used for Sharpe ratio below)
# anchored on treasury bond rates
risk_free_rate = (yf.Ticker('^TNX').info['regularMarketPreviousClose'] / 100)


def display_simulated_portfolios(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    # pull results, weights from random portfolios
    results, weights = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    # pull the max portfolio Sharpe ratio (3rd element in results array from
    # generate_random_portfolios function)
    max_sharpe_idx = np.argmax(results[2])

    # pull the associated standard deviation, annualized return w/ the max Sharpe ratio
    stdev_portfolio, returns_portfolio = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

    # pull the allocation associated with max Sharpe ratio
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=data.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    print("-" * 100)
    print("Portfolio at Maximum Sharpe Ratio\n")
    print("--Returns, Volatility--\n")
    rounded_returns_port = round(returns_portfolio, 2)
    print("Annualized Return:", rounded_returns_port)
    rounded_stdev_portfolio = round(stdev_portfolio, 2)
    print("Annualized Volatility:", rounded_stdev_portfolio)
    print("\n")
    print("--Allocation at Max Sharpe Ratio--\n")
    print(max_sharpe_allocation)
    print("-" * 100)
    plt.figure(figsize=(14, 7.5))

    # x = volatility, y = annualized return, color mapping = sharpe ratio
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.5)
    plt.colorbar()
    ax = plt.gca()

    # Mark the portfolio w/ max Sharpe ratio
    plt.scatter(stdev_portfolio, returns_portfolio, marker='x', color='r', s=100, label='Max Sharpe Ratio')
    plt.scatter(min(results[0]), min(results[1]), marker='x', color='black', s=100, label='Min Variance Portfolio')
    plt.title('Simulated Portfolios Illustrating Efficient Frontier')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Returns')
    plt.text(.01, .75, f'Annualized Return: {rounded_returns_port * 100}\n'
                       f'Annualized Volatility: {rounded_stdev_portfolio * 100}\n \n'
                       f'{max_sharpe_allocation}', transform=ax.transAxes)
    plt.legend(labelspacing=1.2)
    plt.show()


display_simulated_portfolios(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
