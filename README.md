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
<font color="red"> import numpy as np

I = 100000 # number of iteration

z = np.random.standard_normal(I)
ST = S0 * np.exp((r-0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
hT = np.maximum (ST - K, 0)
C0 = np.exp(-r * T) * sum(hT) / I
```

Print the Results:
```
print ("Value of the European Call Option %5.3f%" C0)
```
> Value of the European Call Option 4.080

# Two Approaches for getting Sigma

When talking about volatility, people often relate to “realized volatility” and “implied volatility.” The former is calculated from return data, which reflects more of the past and current status, whereas the latter is calculated from options data, which reflects more on the investor’s expectations for the future.

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
![dd](https://user-images.githubusercontent.com/33922653/33764540-19b272ee-dc58-11e7-83c4-cd2a779e6b22.png)

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
Source from [This Site] (http://webcache.googleusercontent.com/search?q=cache:29VTaDw1YmMJ:www.kafo.or.kr/board_common/file_download.asp%3FBoard_Key%3D351%26File_Key%3D376%26flag%3D1+&cd=1&hl=ko&ct=clnk&gl=kr)


```
from pandas_datareader import Options
from matplotlib.finance import quotes_historical_yahoo_ochl
import matplotlib.pyplot as plt
import datetime

# Step 1: define two functions
def call_data(tickrr,exp_date):
    x = Options(ticker,'yahoo')
    data= x.get_call_data(expiry=exp_date)
    return data
def implied_vol_call_min(S,X,T,r,c):
    from scipy import log,exp,sqrt,stats
    implied_vol=1.0
    min_value=1000
    for i in range(10000):
        sigma=0.0001*(i+1)
        d1=(log(S/X)+(r+sigma*sigma/2.)*T)/(sigma*sqrt(T))
        d2 = d1-sigma*sqrt(T)
        c2=S*stats.norm.cdf(d1)-X*exp(-r*T)*stats.norm.cdf(d2)
        abs_diff=abs(c2-c)
        if abs_diff<min_value:
            min_value=abs_diff
            implied_vol=sigma
            k=i
    return implied_vol

# Step 2: input area
ticker='^GSPC'
exp_date=datetime.date(2017,12,8) # first try not exact
r=0.0003 # estimate
begdate=datetime.date(2010,1,1) # this is arbitrary since we care about current price

# Step 3: get call option data
calls=call_data(ticker,exp_date)
exp_date0=int('20'+calls.Symbol[0][len(ticker):9]) # find exact expiring date
today=datetime.date.today()
p = quotes_historical_yahoo_ochl(ticker, begdate, today, asobject=True, adjusted=True)
s=p.close[-1] # get current stock price
y=int(exp_date0/10000)
m=int(exp_date0/100)-y*100
d=exp_date0-y*10000-m*100
exp_date=datetime.date(y,m,d) # get exact expiring date
T=(exp_date-today).days/252.0 # T in years

# Step 4: run a loop to estimate the implied volatility
n=len(calls.Strike) # number of strike
strike=[] # initialization
implied_vol=[] # initialization
call2=[] # initialization
x_old=0 # used when we choose the first strike
for i in range(n):
    x=calls.Strike[i]
    c=(calls.Bid[i]+calls.Ask[i])/2.0
    if c >0:
        print ('i=',i,', c=',c)
        if x!=x_old:
            vol=implied_vol_call_min(s,x,T,r,c)
            strike.append(x)
            implied_vol.append(vol)
            call2.append(c)
            print (x,c,vol)
            x_old=x

# Step 5: draw a smile
plt.plot(strike,implied_vol,'o')
plt.title('Skewness smile (skew)')
plt.xlabel('Exercise Price')
plt.ylabel('Implied Volatility')
```

Pictures of Examples using KOSPI 200 Options

### Volatility Smile

As in stock or foreign exchange markets, you will notice the so-called
volatility smile, which is most pronounced for the shortest maturity and which becomes
a bit less pronounced for the longer maturities:

![1](https://user-images.githubusercontent.com/33922653/33751221-d2d57726-dc1c-11e7-9256-459114c2031b.png)

### Volatility Term Structure

Taking VIX as an example:
It suggests the market’s expectation on the future volatility. Since volatility is a measure of systematic risk, the VIX term structure suggests the trend of future market risk. If the VIX is upward-sloping, it implies that investors expect to see the volatility (risk) of the market going up in the future. 

If the VIX is downward sloping, it indicates that investors expect to see the volatility of the market going down in the future

![2](https://user-images.githubusercontent.com/33922653/33751229-e239da7c-dc1c-11e7-9894-202a342abfe7.png)

### Volatility Surface

The volatility surface is a three-dimensional plot of stock option implied volatility seen to exist due to discrepancies with how the market prices stock options and what stock option pricing models say that the correct prices should be. 

![3](https://user-images.githubusercontent.com/33922653/33751240-f2069620-dc1c-11e7-8a21-fb755446a56b.png)



#### BSM Assumptions

1. The underlying stock does not pay a dividend and never will.

2. The option must be European-style.

3. Financial markets are efficient.

4. No commissions are charged on the trade.

5. Interest rates remain constant.

6. The underlying stock returns are log-normally distributed.

Read more: The Volatility Surface Explained | Investopedia https://www.investopedia.com/articles/stock-analysis/081916/volatility-surface-explained.asp#ixzz50dq1TrHA
