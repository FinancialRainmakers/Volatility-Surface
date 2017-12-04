# Why Python?

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
ST = S0 * exp(r-05 * sigma ** 2) * T + sigma * sqrt(T) * z)
hT = maximum (ST - K, 0)
C0 = exp(-r * T) * sum(hT) / I
```

Print the Results:
```
print ("Value of the European Call Option %5.3f % C0")
```



## Historical Volatility

## Implied Volatility

### Volatility Smile

### Volatility Term Structure

### Volatility Surface
