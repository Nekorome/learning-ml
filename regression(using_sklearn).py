import pandas as pd 
import quandl 
import math , datetime
import numpy as np
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#quandl.ApiConfig.api_key = ''
df = quandl.get('WIKI/GOOGL')
 
print(df.head())
print("hi")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] *100
df['per-ch'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100

df = df[['Adj. Close','HL_PCT','per-ch','Adj. Volume' ]]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out= int(math.ceil(0.01*len(df)))
print(forecast_out)

df['lable'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['lable'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['lable'])

X_train,X_test,y_train,y_test, = train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test , y_test )
forcast_set = clf.predict(X_lately)


print(accuracy  )
print(forcast_set)

df['Forecast'] = np.nan

# for getting the date on the graph 

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
oneday = 86400
next_unix = last_unix + oneday


for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += oneday
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()  # showing graph

print(df.head)
# forecast at the end is the prediction