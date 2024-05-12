
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import ApproxInference


# Import data, make a copy of the original
df0 = pd.read_csv('SPRICE_Norwegian_Maritime_Data.csv')
# Get characteristics of dataset including columns with missing data as well:
""" dfc1.info()
dfc1.describe() """

# Create new df with variables we want to work with:
# Chosen using pearson correlation coeff
# See figure and excel set up
new_cols = ['Air_temp_Act', 'Rel_Humidity_act', 'Wind_Speed_avg', 'Wind_Direction_vct', 'Precipitation_Type', 'Precipitation_Intensity']
df = df0[new_cols]

unique_fields = df['Precipitation_Type'].unique().tolist()
print("Unique fields:\n", unique_fields)

print(df)

num_stdv = 1

# Define the labels dictionary
labels = {
    'Air_temp_Act': ['low', 'mid', 'high'], 
    'Rel_Humidity_act': ['low', 'mid', 'high'], 
    'Wind_Speed_avg': ['low', 'mid', 'high'], 
    'Wind_Direction_vct': ['low', 'mid', 'high'], 
    'Precipitation_Type': ['low', 'mid', 'high'], 
    'Precipitation_Intensity': ['low', 'mid', 'high']
} 

# Change values in columns temp_max, temp_min and wind according to the boundaries. The change should be according to the labels.
def label_data(row, column, bounds, labels):
    if row[column] < bounds[column][0]:
        return labels[column][0]
    if row[column] > bounds[column][2]:
        return labels[column][2]
    return labels[column][1]

ata_mean, ata_stdv = df['Air_temp_Act'].mean(), df['Air_temp_Act'].std()
rha_mean, rha_stdv = df['Rel_Humidity_act'].mean(), df['Rel_Humidity_act'].std()
ws_mean, ws_stdv = df['Wind_Speed_avg'].mean(), df['Wind_Speed_avg'].std()
wdc_mean, wdc_stdv = df['Wind_Direction_vct'].mean(), df['Wind_Direction_vct'].std()
pt_mean, pt_stdv = df['Precipitation_Type'].mean(), df['Precipitation_Type'].std()
pi_mean, pi_stdv = df['Precipitation_Intensity'].mean(), df['Precipitation_Intensity'].std()

bounds = {
    'Air_temp_Act': [ata_mean - num_stdv * ata_stdv, ata_mean, ata_mean + num_stdv * ata_stdv],
    'Rel_Humidity_act': [rha_mean - num_stdv * rha_stdv, rha_mean, rha_mean + num_stdv * rha_stdv],
    'Wind_Speed_avg': [ws_mean - num_stdv * ws_stdv, ws_mean, ws_mean + num_stdv * ws_stdv],
    'Wind_Direction_vct': [wdc_mean - num_stdv * wdc_stdv, wdc_mean, wdc_mean + num_stdv * wdc_stdv],
    'Precipitation_Type': [pt_mean - num_stdv * pt_stdv, pt_mean, pt_mean + num_stdv * pt_stdv],
    'Precipitation_Intensity': [pi_mean - num_stdv * pi_stdv, pi_mean, pi_mean + num_stdv * pi_stdv]
} 

# Apply the labeling function to the 'precipitation', 'temp_max', 'temp_min', and 'wind' columns
for key, value in labels.items():
    df[key] = df.apply(label_data, args=(key, bounds, labels), axis=1)


""" 

Need to look at the data in Precipitation_Type and Precipitation_Intensity

 """










print(df.describe())

# Let's show all columns with missing data as well:
df[df.isnull().any(axis=1)] # any missing data in columns
df.isnull().any()

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
# The ground work of chosing paramteres (nodes) for the Bayesian Network