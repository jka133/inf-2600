#pip install pgmpy

# Including the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Import data, make a copy of the original
df0 = pd.read_csv('seattle-weather.csv')
dfc1 = df0.copy()
dfc1.head()

# Get characteristics of dataset including columns with missing data as well:
dfc1.info()

# Checking the unique values in the 'weather' column
unique_fields = dfc1['weather'].unique()
print("Unique fields:\n", unique_fields)

dfc1.describe()

# Put categorical varaibles in a list
categorical_lst = ['date','weather']
# Create a seperate & smaller dataframe for categorical variables
dfc2a = pd.DataFrame(dfc1, columns=categorical_lst, copy=True)
dfc2a.head()

# Put all continuous variables into a list
continuous_lst = ['precipitation', 'temp_max', 'temp_min', 'wind']
# Create a seperate & smaller dataframe for our chosen variables. Use 'copy=True' so changes wont affect original
dfc2b = pd.DataFrame(dfc1, columns=continuous_lst, copy=True)
dfc2b.head()

# Create new df with variables we want to work with:
new_cols = ['date', 'precipitation', 'temp_max', 'temp_min', 'wind', 'weather']

df = df0[new_cols]
df.head()

# Let's show all columns with missing data as well:
df[df.isnull().any(axis=1)] # any missing data in columns
df.isnull().any()

num_stdv = 1

# Define the labels dictionary
labels = {

}

# Create bounds for continuous labels


df.head()

# Define the hierarchy
weather_model = BayesianNetwork([
    (categorical_lst), 
    (continuous_lst), 
    (new_cols)
])

# And, the states for each variables

# Calculate Probabilities

# Weather does not have any parents so all we need are the marginal probabilities of observing each weather type

# Joint Propabilities
# Create dict where key=parent, value=child
var_dict = {

            }

# Create conditional distributions and store results in a list
cpd_lst = []
for key, value in var_dict.items():
    pass
    ### Define yourself

# Note that we get 3 Nan values in the above conditional distributions. This is because one of the type of precipitation (low) did not contain any relation with temp_max.
# Therefore, normalization, does not produce the intended result.
# To mitigate this, we replace Nan with the equal probability within the three values, i.e., 0.33
cpd_lst[2][:,0] = .33

cpd_lst

# Creating tabular conditional probability distribution



# Add CPDs and factors to the model


# Check if model is consistent


# Viewing nodes of the model
weather_model.nodes()

# Viewing edges of the model
weather_model.edges()

# Print the probability table of the weather node
print(weather_cpd)

# Print the probability table of the wind node
print(wind_cpd)

# Independcies in the model

# Checking independcies of a particular node


from pgmpy.inference import VariableElimination

# Question 1: (a) What is the probability of high wind when the weather is sunny? (b) What is the probability of sunny weather when the wind is high?


# Question 2:
# (a) Calculate all the possible joint probability and determine the best probable condition. Explain your results?

# (b) What is the most probable condition for precipitation, wind and weather, combined?


# Question 3. Find the probability associated with each weather, given that the precipitation is medium? Explain your result.



# Question 4. What is the probability of each weather condition given that precipitation is medium and wind is low or medium? Explain your method and results. How does the result change with the addition of wind factor compared to question 3 of Task 1.2?



from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling

# Repeat Q.1. (a) of Task 1.2 - What is the probability of high wind when the weather is sunny?



# Repeat Q.1. (b) of Task 1.2 - What is the probability of sunny weather when the wind is high?



# Repeat Q.2 . (a) of Task 1.2 - Calculate all the possible joint probability and determine the best probable condition. Explain your results?



# Repeat Q.2 . (b) of Task 1.2 - What is the most probable condition for precipitation, wind and weather, combined?



from pgmpy.inference import ApproxInference

# Repeat Q.3 of Task 1.2 - Find the probability associated with each weather, given that the precipitation is medium? Explain your result.



# Repeat Q.4 of Task 1.2 - What is the probability of each weather condition given that precipitation is medium and wind is low or medium? Explain your method and results. How does the result change with the addition of wind factor compared to question 3 of Task 1.2?


