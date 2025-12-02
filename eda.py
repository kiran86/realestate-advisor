import pandas as pd
df = pd.read_csv('india_housing_prices.csv')
print(df.head())
print(df.info())
print(df.describe())