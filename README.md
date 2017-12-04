# Why Python?

Source from [This Site] (https://github.com/jasonstrimpel/PyData-Meetup/blob/master/Equity%20Option%20Implied%20Volatility%20Analytics%20with%20Python.ipynb)
Python has become an increasingly important tool in the domain of quantitative and algorithmic trading and research. Users range from senior quantitative analysts and researchings pricing complex derivatives using numerical techniques all the way to the retail trader just getting started. This talk will cover the basics of pricing "plain vanilla" options on listed equities and dive into some analysis of the unobserved feature of listed equity options known as implied volatility.

In this talk, we'll learn a bit about Black-Scholes model, the derived option pricing formula and the "greeks" and how to code it all in Python. I'll then demonstrate how to gather options data using Pandas and apply various transformations to obtain the theoretical value of the option and the associated greeks. We'll then extend the talk to discuss implied volatility and show how to use Numpy methods to compute implied volatility and model missing and bad values using interpolation. Finally, we'll use the results to visualize the so-called volatility skew and term structure to help inform potential trading decisions.

# Getting Option Values
```
import numpy as np
import pandas as pd
import pandas.to.data as web

goog = web.DataReader('GOOG', data_source = 'google', start = '3/14/2009', and '4/14/2014')


```


To learn Coding

## Monte Carlo Simulation for Options
Suppose we have the following numerical parameter values for the valuation:

- Initial Stock Index Level S0 = 100
- Strike Price of the European Call Option K = 105
- Time-to-maturity T = 1 Year
- Constant, riskless short rate r = 5%
- Constant Volatility sigma = 20%

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
import numpy as np
import pandas as pd
from pandas_datareader import data

KS200 = data.DataReader("KRX:KOSPI200", "google")
KS200.head()

KS200['Log_Ret'] = np.log(KS200['Close'] / KS200['Close'].shift(1))
KS200['Volatility'] = pd.rolling_std(KS200['Log_Ret'],
window=252) * np.sqrt(252)

KS200[['Close', 'Volatility']].plot(subplots=True, color='blue',
figsize=(8, 6))
```

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
