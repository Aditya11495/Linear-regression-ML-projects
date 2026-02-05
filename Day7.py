import pandas as pd
df=pd.read_csv("C:\PYTHON\yahoo_finance_template.csv")

df = df.drop('Adj Close',axis=1) #to drop a column

df=df.fillna(157.8) #to fill the missing data with any value

df.loc[df['Open'] < 0, 'Open'] = 0  #to replace negative values with 0
df.loc[df['Open'] > 100, 'Open'] = 100     #to replace values based on condition

import pandas as pd
import matplotlib.pyplot as plt 

df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(8,5))

# High-Low line
plt.vlines(df['Date'], df['Low'], df['High'])

# Open price
plt.scatter(df['Date'], df['Open'], label='Open')

# Close price
plt.scatter(df['Date'], df['Close'], label='Close')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock OHLC Representation')
plt.legend()
plt.grid(True)
plt.show()

print(df)
