# Options Greeks

This method calculates an options' greeks 

An option's price can be influenced by a number of factors that can either help or hurt traders depending on the type of positions they have taken. Successful traders understand the factors that influence options pricing, which include the so-called "Greeks"—a set of risk measures so named after the Greek letters that denote them, which indicate how sensitive an option is to time-value decay, changes in implied volatility, and movements in the price its underlying security.

These four primary Greek risk measures are known as an option's theta, vega, delta, and gamma. In  this method we will calculate Gamma,Delta and Vega. 

Delta: Delta is a measure of the change in an option's price (that is, the premium of an option) resulting from a change in the underlying security.
You can think of it as the first derivative of options premium with the underlying asset's price

Gamma: Gamma measures the rate of changes in delta over time. Since delta values are constantly changing with the underlying asset's price, gamma is used to measure the rate of change and provide traders with an idea of what to expect in the future.
You can think of it as the second derivative of options premium with the uderlying asset's price

Vega: Vega measures the risk of changes in implied volatility or the forward-looking expected volatility of the underlying asset price. While delta measures actual price changes, vega is focused on changes in expectations for future volatility. Higher volatility makes options more expensive since there’s a greater likelihood of hitting the strike price at some point.
You can think of it as the first derivative of options premium with the return of the underlying asset

If you want to understand better greeks you can visit : https://www.investopedia.com/trading/getting-to-know-the-greeks/
