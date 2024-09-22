import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize

data = pd.read_excel('Data_190324.xlsx', sheet_name=None)
dataMacro = data['Macro']
dataPrices = data['Prices']

dataYield = data['Yield']
tempDF = dataMacro.merge(dataPrices, on='Date')
mergedDF = (tempDF.merge(dataYield, on='Date')).dropna()

kmeans = KMeans(n_clusters=3, random_state = 33)
mergedDF['Regime'] = kmeans.fit_predict(mergedDF[['GDP YOY', 'CPI YOY']])
mergedDF.to_csv('output.csv', index=False)
centroids = kmeans.cluster_centers_
print(centroids)

mergedDF['S&P 500 Returns'] = mergedDF['S&P 500'].pct_change()
mergedDF['Gold Returns'] = mergedDF['Gold'].pct_change()
mergedDF['USD Index Returns'] = mergedDF['USD Index Spot Rate'].pct_change()
mergedDF['US 10YR Returns'] = mergedDF['US 10YR Bonds'].pct_change()
mergedDF = mergedDF.dropna()

def portfolioVariance(weights, goldReturns, usdIndexReturns, us10yrReturns):
    returns = weights[0]*goldReturns + weights[1]*usdIndexReturns + weights[2]*us10yrReturns
    variance = np.var(returns)
    return variance

for regime in mergedDF['Regime'].unique():
    regime_data = mergedDF[mergedDF['Regime'] == regime]
    initial_weights = [1/3, 1/3, 1/3] #initial 
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1), (0, 1), (0, 1)]
    result = minimize(portfolioVariance, initial_weights, args=(regime_data['Gold Returns'], regime_data['USD Index Returns'], regime_data['US 10YR Returns']), method='SLSQP', bounds=bounds, constraints=constraints)
    print(f'Optimal weights for regime {regime}: {result.x}')
