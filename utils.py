"""Utility functions:
- Hill estimator 
- Cross-correlation function
- GARCH(1,1) model
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

def hill_estimator(sample: pd.Series, m: int) -> float:
    """Given a pandas Series `sample` representing samples of a random variable X, return the Hill estimate for the tail index of X, computed as the empirical mean of the values log(x_i / x_(m+1)) for i=1,...,m, where `x_i` denotes the (i-1)st largest value in `sample`.
    
    Parameters
    ----------
    sample: Pandas Series of floats or ints.
    m: Int < sample.size indicating the number of upper order statistics to use.

    Returns
    -------
    Float corresponding to Hill estimate of the sample using the first m upper order statistics.
    """
    if m >= sample.size: 
        raise ValueError('The parameter \'m\' corresponds to the number of upper order statistics from your sample used in computing the Hill estimate, and thus should be an integer less than the length of your series.') 
    order_stats = sample.sort_values(ascending=False)
    Xmin = order_stats.iloc[m]
    order_stats = order_stats.iloc[:m]
    gamma = 1/m * np.log(order_stats / Xmin).sum()
    return 1/gamma

def cross_corr(S_1: pd.Series, S_2: pd.Series, max_lags: int = None, match_indices: bool = True) -> pd.Series:
    """Given pandas Series `S_1` and `S_2` of floats or ints, return a Series of length `max_lags` whose entry at index i is the autocorrelation function of S_1 and S_2 with lag i. 
    
    Parameters
    ----------
    S_1: Pandas Series of floats or ints.
    S_2: Pandas Series of floats or ints.
    max_lags: Int corresponding to the maximum number of lags to compute. If `None`, compute the maximum possible number of lags (given by min(S_1.size,S_2.size)).
    m: int <= 
    

    Returns
    -------
    Pandas Series
    """
    if match_indices == True:
        index = S_1.index.intersection(S_2.index)
        S_1 = S_1[index]
        S_2 = S_2[index]
        n = S_1.size
    else: 
        n = min(S_1.size, S_2.size)
        S_1 = S_1.iloc[:n]
        S_2 = S_2.iloc[:n]
    if not max_lags:
        max_lags = n-1
    elif max_lags > min(S_1.size, S_2.size):
        raise ValueError('\'max_lags\' is too large.')
    sigma_1 = S_1.std()
    sigma_2 = S_2.std()
    mu_1 = S_1.mean()
    mu_2 = S_2.mean()
    rho = []
    S_1 = S_1.reset_index(drop=True)
    S_2 = S_2.reset_index(drop=True)
    for l in range(0, max_lags+1):
        S_lead = S_1[l:].reset_index(drop=True)
        S_lag = S_2[:n-l]
        rho_l = np.sum((S_lead-mu_1)*(S_lag-mu_2))
        rho.append(rho_l/(n*sigma_1*sigma_2))
    return np.array(rho)

def uncalibrated_GARCH(params: list, var_0: float, ser: pd.Series) -> pd.Series:
    """Given a list of paramater variables [alpha_0, alpha_1, beta_0], an initial estimate sigma^2_1 = `var_0`, and a series `ser` of asset return innovations, return a series of the GARCH(1,1) volatility 
    
    sigma_t^2 = alpha_0 + alpha_1 ser.iloc[t-1]^2 + beta_0 sigma_{t-1}
    
    of the return process.
    
    Parameters
    ----------
    params: List of floats [alpha_0, alpha_1, beta_0] corresponding to the parameters of the GARCH(1,1) model.
    
    var_0: Float corresponding to the initial value sigma_t^2 of the 
    
    ser: Pandas Series of floats corresponding to an asset return innovation series 
    
    m_t = r_t - E[r_t|F_{t-1}]. 

    Returns
    -------
    Pandas series of floats with index ser.index corresponding to the GARCH(1,1) volatility of the series `ser`.
    """
    [alpha_0, alpha_1, beta] = params
    var_t = [var_0]
    for t in range(1, ser.size):
        prev_var = var_t[t-1]
        var_t.append(alpha_0 + alpha_1 * ser.iloc[t-1] ** 2 + beta * prev_var)
    return pd.Series(var_t, index=ser.index, name='GARCH (in-sample)')

def GARCH(params: list, var_0: float, index: pd.DatetimeIndex) -> pd.Series:
    """Given a list of paramater variables [alpha_0, alpha_1, beta_0], an initial estimate sigma^2_1 = `var_0`, and an pd.DatetimeIndex `index`, return a series of the GARCH(1,1) volatility series
    
    sigma_t^2 = alpha_0 + alpha_1 m_t^2 + beta_0 sigma_{t-1}
    
    over the DatetimeIndex `index`, where the innovation sequence follows a standard normal, m_t ~ Norm(0,1).
    
    Parameters
    ----------
    params: List of floats [alpha_0, alpha_1, beta_0] corresponding to the parameters of the GARCH(1,1) model.
    
    var_0: Float corresponding to the initial value sigma_t^2 of the 
    
    index: Pandas DatetimeIndex corresponding to the index of the return innovation series.

    Returns
    -------
    Pandas series of floats with index ser.index corresponding to the GARCH(1,1) volatility of the innovation sequence following a standard normal distribtuion.
    """
    [alpha_0, alpha_1, beta] = params
    var_t = [var_0]
    for t in range(1, index.size):
        prev_var = var_t[t-1]
        var_t.append(alpha_0 + alpha_1 * prev_var* stats.norm.rvs()** 2 + beta * prev_var)
    return pd.Series(var_t, index=index, name='GARCH (out-of-sample)') 