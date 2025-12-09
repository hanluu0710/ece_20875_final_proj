import pandas
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
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

def getRelevant():
    for i in range(4):
        x = np.column_stack((dataset_2["Brooklyn Bridge"], dataset_2['Manhattan Bridge'], dataset_2['Queensboro Bridge'], dataset_2['Williamsburg Bridge']))
        linear_reg = LinearRegression().fit(x, dataset_2["Total"])
        print(linear_reg.coef_)

getRelevant()