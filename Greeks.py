import numpy as np
from scipy import stats

def BlackScholesPrice (S,K,r,sigma,d,t,optionType):
    '''
    function that calculates European's option price using
    Black and Scholes method

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    K : float
        Strike price.
    r : float
        Return of underlying asset.
    sigma : float
        Standard Deviation of underlying asset.
    d : float
        Dividend return.
    t : float
        Maturity of Options.
    optionType : string
        Call or Put.

    Returns
    -------
    P : float
        Black and Scholes Price of a call or put.

    '''
    d1 = (np.log(S/K) + (r-d + np.power(sigma,2)/2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if optionType == "CALL" or optionType == "C" or optionType == "call" or optionType == "Call" :
        P =  S*stats.norm.cdf(d1) - K * np.exp(-(r-d)*t) * stats.norm.cdf(d2)
    else:
        P = K * np.exp(-(r-d)*t) * stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
    return P


def MCvanillaOptions (S,K,r,sigma,d,t,dt,inseed,npaths):
    '''This function calculates the price of a vanilla options using Monte 
    Carlo simulation. 
    
    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    K : float
        Strike price.
    r : float
        Return of underlying asset.
    sigma : float
        Standard Deviation of underlying asset.
    d : float
        Dividend return.
    t : float
        Maturity of Options.
    dt : float
        period of asian's option intervals calculation.
    inseed : float
        Initial seed.
    npaths : int
        Number of paths for Monte Carlo simulation

    Returns
    -------
    MCVanillaCall : float
        Call price using Monte Carlo.
    MCVanillaPut : float
        Put price using Monte Carlo.
    Call_Error : float
        Error between the Black and Scholes and Monte Carlo for a Call.
    Put_Error : float
        Error between the Black and Scholes and Monte Carlo for a Put.

    '''
    #We are doing pricing vanilla with the same number of 
    #steps as the Asian in order to not have errors in the calculations
    #That's the reason why as well that we are using a constant seed
    t_obs = np.flip(np.arange(t, 1/365, -dt))
    nsteps = len(t_obs)
    # Generate random numbers using a given seed for the number generator
    np.random.seed(seed=inseed)
    wn = np.random.standard_normal(size=(npaths, nsteps))
    wn2 = stats.zscore(np.block([[wn], [-wn]]))
    # Use Euler's equation to get the next step
    St = np.zeros((2*npaths, nsteps), dtype=float)
    St[:, 0] = S*np.exp((r-d-sigma**2/2)*t_obs[0]+sigma*np.sqrt(t_obs[0])
                        * wn2[:, 0])
    for i in range(nsteps-1):
        if nsteps == 1:
            continue
        elif nsteps > 1:
            dt_ = t_obs[i+1] - t_obs[i]
            St[:, i+1] = St[:, i] * \
                np.exp((r-d-sigma**2/2)*dt_+sigma*np.sqrt(dt_)*wn2[:, i+1])
    # Histogram
    # plt.hist(St, bins=100)
    call_payoffs = np.maximum(St[:, -1]-K, 0)
    put_payoffs = np.maximum(K-St[:, -1], 0)
    MCVanillaCall = np.mean(call_payoffs)*np.exp(-r*t)
    MCVanillaPut = np.mean(put_payoffs)*np.exp(-r*t)
    BSCall= BlackScholesPrice(S,K,r,sigma,d,t,"CALL")
    BSPut = BlackScholesPrice(S,K,r,sigma,d,t,"PUT")
    Call_Error = BSCall-MCVanillaCall
    Put_Error = BSPut - MCVanillaPut
    return (MCVanillaCall, MCVanillaPut, Call_Error, Put_Error)


def MCAsianOptions (S,K,r,sigma,d,t,dt,inseed,npaths):
    '''This function calculates the price of a vanilla options using Monte 
    Carlo simulation. 
    
    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    K : float
        Strike price.
    r : float
        Return of underlying asset.
    sigma : float
        Standard Deviation of underlying asset.
    d : float
        Dividend return.
    t : float
        Maturity of Options.
    dt : float
        period of asian's option intervals calculation.
    inseed : float
        Initial seed.
    npaths : int
        Number of paths for Monte Carlo simulation

    Returns
    -------
    Asian_Call : float
        Asian Call price using Monte Carlo.
    Asian_Put : float
        Asian Put price using Monte Carlo.
    t_obs : float
        Schedule of asian's option intervals calculation
    
    '''
    #We are doing pricing vanilla with the same number of 
    #steps as the Asian in order to not have errors in the calculations
    #That's the reason why as well that we are using a constant seed
    t_obs = np.flip(np.arange(t, 1/365, -dt))
    nsteps = len(t_obs)
    # Generate random numbers using a given seed for the number generator
    np.random.seed(seed=inseed)
    wn = np.random.standard_normal(size=(npaths, nsteps))
    # Antithetics
    wn2 = stats.zscore(np.block([[wn], [-wn]]))
    # Use Euler's equation to get the next step
    St = np.zeros((2*npaths, nsteps), dtype=float)
    St[:, 0] = S*np.exp((r-d-sigma**2/2)*t_obs[0]+sigma*np.sqrt(t_obs[0])
                        * wn2[:, 0])
    for i in range(nsteps-1):
        if nsteps == 1:
            continue
        elif nsteps > 1:
            dt_ = t_obs[i+1] - t_obs[i]
            St[:, i+1] = St[:, i] * \
                np.exp((r-d-sigma**2/2)*dt_+sigma*np.sqrt(dt_)*wn2[:, i+1])
    # We compute the arithmetic average per row (path)
    Smean = St.mean(axis=1)
    call_payoffs = np.maximum(Smean-K, 0)
    put_payoffs = np.maximum(K-Smean, 0)
    Asian_Call = np.mean(call_payoffs)*np.exp(-r*t)
    Asian_Put = np.mean(put_payoffs)*np.exp(-r*t)
    return (Asian_Call,Asian_Put,t_obs)


def Greeks_Delta (S,K,r,sigma,d,t,dt,inseed,npaths):
    """
    It calculate the Delta of a call and a put using Monte Carlo
    and Black and Scholes

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    K : float
        Strike price.
    r : float
        Return of underlying asset.
    sigma : float
        Standard Deviation of underlying asset.
    d : float
        Dividend return.
    t : float
        Maturity of Options.
    dt : float
        period of asian's option intervals calculation.
    inseed : float
        Initial seed.
    npaths : int
        Number of paths for Monte Carlo simulation

    Returns
    -------
    delta_Call : flaot
        Monte Carlo delta of a Call
    delta_Put : flaot
        Monte Carlo delta of a Put

    """
    ds = S/100
    (Call0, Put0, Call_Error, Put_Error) = \
        MCvanillaOptions(S, K, r, sigma, d, t, dt, inseed, npaths)
    (Callup, Putup, Call_ErrorUp, Put_ErrorUp) =\
        MCvanillaOptions(S+ds, K, r, sigma, d, t, dt, inseed, npaths) 
    (Calldn, Putdn, Call_Errordn, Put_Errordn) =\
        MCvanillaOptions(S-ds, K, r, sigma, d, t, dt, inseed, npaths) 
    delta_Call_up = (Callup-Call0)/ds
    delta_Call_down = (Calldn-Call0)/(-ds)
    delta_Call = (delta_Call_up+delta_Call_down)/2
    delta_Put_up = (Putup-Put0)/ds
    delta_Put_down = (Putdn-Put0)/(-ds)
    delta_Put = (delta_Put_up+delta_Put_down)/2
    return (delta_Call,delta_Put)


def Greeks_Gamma (S,K,r,sigma,d,t,dt,inseed,npaths):
    """
    It calculate the Gamma of a call and a put using Monte Carlo
    and Black and Scholes

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    K : float
        Strike price.
    r : float
        Return of underlying asset.
    sigma : float
        Standard Deviation of underlying asset.
    d : float
        Dividend return.
    t : float
        Maturity of Options.
    dt : float
        period of asian's option intervals calculation.
    inseed : float
        Initial seed.
    npaths : int
        Number of paths for Monte Carlo simulation

    Returns
    -------
    Gamma : flaot
        Monte Carlo gamma of an options

    """
    ds = S/100
    (Call0, Put0, Call_Error, Put_Error) = \
        MCvanillaOptions(S, K, r, sigma, d, t, dt, inseed, npaths)
    (Callup, Putup, Call_ErrorUp, Put_ErrorUp) =\
        MCvanillaOptions(S+ds, K, r, sigma, d, t, dt, inseed, npaths) 
    (Calldn, Putdn, Call_Errordn, Put_Errordn) =\
        MCvanillaOptions(S-ds, K, r, sigma, d, t, dt, inseed, npaths) 
    delta_Call_up = (Callup-Call0)/ds
    delta_Call_down = (Calldn-Call0)/(-ds)
    gamma = (delta_Call_up-delta_Call_down)/ds
    return (gamma)


def Greeks_Vega (S,K,r,sigma,d,t,dt,inseed,npaths):
    """
    It calculate the Vega of a call and a put using Monte Carlo
    and Black and Scholes

    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    K : float
        Strike price.
    r : float
        Return of underlying asset.
    sigma : float
        Standard Deviation of underlying asset.
    d : float
        Dividend return.
    t : float
        Maturity of Options.
    dt : float
        period of asian's option intervals calculation.
    inseed : float
        Initial seed.
    npaths : int
        Number of paths for Monte Carlo simulation

    Returns
    -------
    Vega : flaot
        Monte Carlo Vega of an options

    """
    dvol = 0.01
    (Call0, Put0, Call_Error, Put_Error) = \
        MCvanillaOptions(S, K, r, sigma, d, t, dt, inseed, npaths)
    (Callup, Putup, Call_ErrorUp, Put_ErrorUp) =\
        MCvanillaOptions(S, K, r, sigma+dvol, d, t, dt, inseed, npaths) 
    (Calldn, Putdn, Call_Errordn, Put_Errordn) =\
        MCvanillaOptions(S, K, r, sigma+dvol, d, t, dt, inseed, npaths) 
    vega_Call_up = (Callup-Call0)/dvol
    vega_Call_down = (Calldn-Call0)/(-dvol)
    vega = (vega_Call_up+vega_Call_down)/2
    return (vega)


def BSGreeks (S,K,r,sigma,d,t,dt,inseed,npaths,SensType):
    """
    It calculates the sensitivity measurement given a specific sensitivity 
    name using Black Scholes
    
    Parameters
    ----------
    S : float
        Spot price of underlying asset.
    K : float
        Strike price.
    r : float
        Return of underlying asset.
    sigma : float
        Standard Deviation of underlying asset.
    d : float
        Dividend return.
    t : float
        Maturity of Options.
    dt : float
        period of asian's option intervals calculation.
    inseed : float
        Initial seed.
    npaths : int
        Number of paths for Monte Carlo simulation.
    SensType: string
        It takes the name of the sensitivity measure

    Returns
    -------
    BSSensitivity_call : float
        Call's sensitivity.
    BSSensitivity_put : float
        Put's sensitivity.

    """
    d1 = (np.log(S/K)+(r-d+sigma**2/2)*t)/(sigma*np.sqrt(t))
    if SensType == "DELTA" or SensType == "D" or SensType == "d" or SensType == "delta":
        BSdelta_Call = stats.norm.cdf(d1)
        BSdelta_Put = BSdelta_Call - 1
        return (BSdelta_Call,BSdelta_Put)
    elif SensType == "GAMMA" or SensType == "G" or SensType == "g" or SensType == "gamma":
        BSGamma = stats.norm.pdf(d1)/(S*sigma*np.sqrt(t))
        return (BSGamma,BSGamma)
    else:
        BSVega = S*stats.norm.pdf(d1)*np.sqrt(t)
        return (BSVega,BSVega)
    
        
    
vol = 0.2
r = 0.025
d = 0.01
t = 9.0/12
s0 = 11
k = 12
npaths = 100000
inseed = 1312 #A random number
dt = 1.0/12

(MCVanillaCall, MCVanillaPut, Call_Error, Put_Error) = \
    MCvanillaOptions(s0,k,r,vol,d,t,dt,inseed,npaths)

(Asian_Call,Asian_Put,t_obs) = \
     MCAsianOptions(s0,k,r,vol,d,t,dt,inseed,npaths)
     
print("Vanilla Call/Put: ",MCVanillaCall+Call_Error," , ",MCVanillaPut+Put_Error)
print("Asian Call/Put: ",Asian_Call+Call_Error," , ",Asian_Put+Put_Error)

(delta_Call,delta_Put) = \
    Greeks_Delta(s0, k, r, vol, d, t, dt, inseed, npaths)

(BSdelta_Call,BSdelta_Put) = BSGreeks(s0,k,r,vol,d,t,dt,inseed,npaths,"d")

delta_call_error = BSdelta_Call - delta_Call
delta_put_error = BSdelta_Put - delta_Put

print("Delta Call/Put: ",delta_Call+delta_call_error," , ",delta_Put+delta_put_error)

gamma = Greeks_Gamma(s0, k, r, vol, d, t, dt, inseed, npaths)
(BSgamma,BSgamma0) = BSGreeks(s0,k,r,vol,d,t,dt,inseed,npaths,"g")

gamma_error = BSgamma - gamma
print("Gamma options: ",gamma+gamma_error)

vega =  Greeks_Vega(s0, k, r, vol, d, t, dt, inseed, npaths)
(BSvega,BSvega0) = BSGreeks(s0,k,r,vol,d,t,dt,inseed,npaths,"v")

vega_error = BSvega - vega
print("Vega options: ",vega+vega_error)













