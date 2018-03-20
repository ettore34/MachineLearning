import pandas as pd 
import quandl, math, datetime
import numpy as np
#support vector machine
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
#store the trained data 
import pickle
#to plot 
import matplotlib.pyplot as plt 
# pretty style
from matplotlib import style 
# which style 
style.use('ggplot')



# api key GzGRqBs8sn3-DmGFw39Y

#regression Turtorial
df = quandl.get('WIKI/GOOGL')

#print(df.head())

df =df[['Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0

df= df[['Adj. Close', 'HL_PCT','PCT_change','Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-999, inplace = True ) 

# 0.01 is predicting the data from the last  30 days 
forecast_out = int(math.ceil(0.01*len(df)))

print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out) 


#learning data 
X = np.array(df.drop(['label'],1) )
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace = True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.02)

#clf is = to the algorithm chosen 
#clf = svm.SVR() <- LimitExceededError:
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#--1 all CPUs are used.
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
 
#commented out sincce data is up to date trained. 
#should be trained at least once a week 
# #after train we can use pickle to used the saved data 
#with open('linearregession.pickle', 'wb') as f: 
    ##dump into pickle\
#    pickle.dump(clf,f)


#store clf() with the trained data stored in pickle 
pickle_in = open('linearregession.pickle', 'rb') 
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test,y_test)

forecast_set = clf.predict(X_lately)


print("*************after change******************")
print("last days ")
print(df.tail())
print("forecast next ",forecast_out," days",   forecast_set, " accuracy " , accuracy)




# using graph 
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)  
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()    
df['Forecast'].plot() 
plt.legend(loc=4 )
plt.xlabel('Date') 
plt.ylabel('Price')
plt.show()
#print(len(X),len(y))





#print(df.head())
#print(df.tail())
