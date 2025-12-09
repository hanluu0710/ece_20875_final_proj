import pandas
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('nyc_bicycle_counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']                = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
dataset_2['High Temp']            = pandas.to_numeric(dataset_2['High Temp'].replace(',','', regex=True))
dataset_2['Low Temp']             = pandas.to_numeric(dataset_2['Low Temp'].replace(',','', regex=True))
dataset_2['Precipitation']         = pandas.to_numeric(dataset_2['Precipitation'].replace(',','', regex=True))

x_samples = [dataset_2["Brooklyn Bridge"], dataset_2['Manhattan Bridge'], dataset_2['Queensboro Bridge'], dataset_2['Williamsburg Bridge']]
x_labels = ["Brooklyn Bridge", 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']



def getRelevant():
    global x_samples
    scores = []
    
    for i in range(4):
        x = np.column_stack(x_samples[1:])
        x_samples = x_samples[1:] + x_samples[:1]
        linear_model = LinearRegression().fit(x, dataset_2['Total'])
        scores.append(linear_model.score(x, dataset_2['Total']))

    print(scores)
    print(f"Sensor should not be on {x_labels[scores.index(min(scores))]}")

def weather (dataset_2):
    dataset_2["Total"]  = pandas.to_numeric(dataset_2["Total"].replace(',','', regex=True))
    dataset_2["Precipitation"]  = pandas.to_numeric(dataset_2["Precipitation"].replace(',','', regex=True))
    dataset_2["Low Temp"]  = pandas.to_numeric(dataset_2["Low Temp"].replace(',','', regex=True))
    dataset_2["High Temp"]  = pandas.to_numeric(dataset_2["High Temp"].replace(',','', regex=True))


    X = dataset_2[["High Temp", "Low Temp", "Precipitation"]]
    y = dataset_2["Total"]


    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)

    print("R2:", r2_score(y_test, preds))


getRelevant()
weather(dataset_2)