# Why Python?

Source from [This Site] (https://github.com/jasonstrimpel/PyData-Meetup/blob/master/Equity%20Option%20Implied%20Volatility%20Analytics%20with%20Python.ipynb)

Python has become an increasingly important tool in the domain of quantitative and algorithmic trading and research. Users range from senior quantitative analysts and researchings pricing complex derivatives using numerical techniques all the way to the retail trader just getting started. This talk will cover the basics of pricing "plain vanilla" options on listed equities and dive into some analysis of the unobserved feature of listed equity options known as implied volatility.

In this talk, we'll learn a bit about Black-Scholes model, the derived option pricing formula and the "greeks" and how to code it all in Python. I'll then demonstrate how to gather options data using Pandas and apply various transformations to obtain the theoretical value of the option and the associated greeks. We'll then extend the talk to discuss implied volatility and show how to use Numpy methods to compute implied volatility and model missing and bad values using interpolation. Finally, we'll use the results to visualize the so-called volatility skew and term structure to help inform potential trading decisions.

## Coding Example : Drawing Apple Stock Prices
```
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2016, 12, 31)

df = web.DataReader('AAPL', "yahoo", start, end)

print(df.head())

df['Adj Close'].plot()
plt.show()
```
![figure_1-1](https://user-images.githubusercontent.com/33922653/33591982-06020ba8-d9cc-11e7-800e-a590e82379fa.png)


# Monte Carlo Simulation for Options
Suppose we have the following numerical parameter values for the valuation:

- Initial Stock Index Level S0 = 100
- Strike Price of the European Call Option K = 105
- Time-to-maturity T = 1 Year
- Constant, riskless short rate r = 5%
- Constant Volatility sigma = 10%

Black-Scholes-Merton (1973) index level at maturity

Input Parameters:
```
S0 = 100
K = 105
T = 1.0
r = 0.05
sigma = 0.1
```

Valuation Algorithm:
```
from numpy import *

I = 100000 # number of iteration

z = random.standard_normal(I)
ST = S0 * exp((r-0.5 * sigma ** 2) * T + sigma * sqrt(T) * z)
hT = maximum (ST - K, 0)
C0 = exp(-r * T) * sum(hT) / I
```

Print the Results:
```
print ("Value of the European Call Option %5.3f%" C0)
```
> Value of the European Call Option 4.080

# Two Approaches for getting Sigma

## Historical Volatility


```
from __future__ import division
from pandas_datareader import data
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
 
# get stock ticker symbol from user
stock_symbol = "^GSPC"
 
returns = []; stds = []
for days in [31, 92, 365]:
    # set time period for historical prices
    end_time = datetime.today(
        ).strftime('%m/%d/%Y')  # current date
    start_time = (datetime.today() -
        timedelta(days=days)
        ).strftime('%m/%d/%Y')
 
    # retreive historical prices for stock
    prices = data.DataReader(stock_symbol,
        data_source='yahoo',
        start=start_time, end=end_time)
 
    # sort dates in descending order
    prices.sort_index(ascending=False, inplace=True)
 
    # calculate daily logarithmic return
    prices['Return'] = (np.log(prices['Close'] /
        prices['Close'].shift(-1)))
 
    # calculate daily standard deviation of returns
    d_std = np.std(prices.Return)
 
    # annualize daily standard deviation
    std = d_std * 252 ** 0.5
 
    returns.append(list(prices.Return))
    stds.append(std)
 
# Plot histograms
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
n, bins, patches = ax.hist(returns[2][:-1],
    bins=50, alpha=0.65, color='blue',
    label='12-month')
n, bins, patches = ax.hist(returns[1][:-1],
    bins=50, alpha=0.65, color='green',
    label='3-month')
n, bins, patches = ax.hist(returns[0][:-1],
    bins=50, alpha=0.65, color='magenta',
    label='1-month')
ax.set_xlabel('log return of stock price')
ax.set_ylabel('frequency of log return')
ax.set_title('Historical Volatility for ' +
    stock_symbol)
 
# get x and y coordinate limits
x_corr = ax.get_xlim()
y_corr = ax.get_ylim() 
 
# make room for text
header = y_corr[1] / 5
y_corr = (y_corr[0], y_corr[1] + header)
ax.set_ylim(y_corr[0], y_corr[1])
 
# print historical volatility on plot
x = x_corr[0] + (x_corr[1] - x_corr[0]) / 30
y = y_corr[1] - (y_corr[1] - y_corr[0]) / 15
ax.text(x, y , 'Annualized Volatility: ',
    fontsize=11, fontweight='bold')
x = x_corr[0] + (x_corr[1] - x_corr[0]) / 15
y -= (y_corr[1] - y_corr[0]) / 20
ax.text(x, y, '1-month  = ' + str(np.round(stds[0], 3)),
    fontsize=10)
y -= (y_corr[1] - y_corr[0]) / 20
ax.text(x, y, '3-month  = ' + str(np.round(stds[1], 3)),
    fontsize=10)
y -= (y_corr[1] - y_corr[0]) / 20
ax.text(x, y, '12-month = ' + str(np.round(stds[2], 3)),
    fontsize=10)
 
# add legend
ax.legend(loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=3, fontsize=11)
 
# display plot
fig.tight_layout()
fig.show()
```
![spyder](https://user-images.githubusercontent.com/33922653/33655141-9e592352-dab5-11e7-8a62-de7d792dc0a8.png)

## Implied Volatility

Black-Scholes-Merton (1973) functions:
```
def bsm_call_value (S0, K, T, r, sigma):
  from math import log, sqrt, exp
  from scipy import stats
  S0 = float(S0)
  d1 = (log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
  d2 = (log(S0/K)+(r-0.5*sigma**2)*T)/(sigma*sqrt(T))
  value = (S0 * stats.norm.cdf(d1,0.0,1.0) - K * exp(-r*T) * stats.norm.cdf(d2,0.0,1.0))
  return (value)
```

Vega functions:
```
def bsm_vega(S0, K, T, r, sigma):
  from math import log, sqrt
  from scipy import stats
  S0 = float(S0)
  d1 = (log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
  vega = S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(T)
  return (vega)
```

Implied Volatility function:

```
def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
  for i in range (it):
    sigma_est -= ((bsm_call_value(S0,K,T,r,sigma_est)-C0) / bsm_vega(S0,K,T,r,sigma_est))
  return (sigma_est)
```


### Volatility Smile

### Volatility Term Structure

### Volatility Surface
