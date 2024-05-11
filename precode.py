#pip install pgmpy

# Including the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD

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
    'precipitation': ['low', 'mid', 'high'], 
    'temp_max': ['low', 'mid', 'high'], 
    'temp_min': ['low', 'mid', 'high'], 
    'wind': ['low', 'mid', 'high'], 
    'weather': ['drizzle', 'rain', 'sun', 'snow', 'fog']
} 

# Create bounds for continuous labels
prec_mean = df0['precipitation'].mean()
prec_stdv = df0['precipitation'].std()

temp_max_mean = df0['temp_max'].mean()
temp_max_stdv = df0['temp_max'].std()

temp_min_mean = df0['temp_min'].mean()
temp_min_stdv = df0['temp_min'].std()

wind_mean = df0['wind'].mean()
wind_stdv = df0['wind'].std()

bounds = {
    'precipitation': [prec_mean - num_stdv * prec_stdv, prec_mean, prec_mean + num_stdv * prec_stdv],
    'temp_max': [temp_max_mean - num_stdv * temp_max_stdv, temp_max_mean, temp_max_mean + num_stdv * temp_max_stdv],
    'temp_min': [temp_min_mean - num_stdv * temp_min_stdv, temp_min_mean, temp_min_mean + num_stdv * temp_min_stdv],
    'wind': [wind_mean - num_stdv * wind_stdv, wind_mean, wind_mean + num_stdv * wind_stdv]
}

# Change values in columns temp_max, temp_min and wind according to the boundaries. The change should be according to the labels.
def label_data(row, column, bounds, labels):
    if row[column] < bounds[column][0]:
        return labels[column][0]
    elif row[column] < bounds[column][1]:
        return labels[column][1]
    else:
        return labels[column][2]

# Apply the labeling function to the 'precipitation', 'temp_max', 'temp_min', and 'wind' columns
df['precipitation'] = df.apply(label_data, args=('precipitation', bounds, labels), axis=1)
df['temp_max'] = df.apply(label_data, args=('temp_max', bounds, labels), axis=1)
df['temp_min'] = df.apply(label_data, args=('temp_min', bounds, labels), axis=1)
df['wind'] = df.apply(label_data, args=('wind', bounds, labels), axis=1)

print(df)

# Define the hierarchy
weather_model = BayesianNetwork([
    ('weather', 'precipitation'),
    ('weather', 'wind'),
    ('precipitation', 'temp_max'),
    ('wind', 'temp_min')
])

# And, the states for each variables
states = {
    'weather': ['drizzle', 'rain', 'sun', 'snow', 'fog'],
    'precipitation': ['low', 'mid', 'high'],
    'temp_max': ['low', 'mid', 'high'],
    'temp_min': ['low', 'mid', 'high'],
    'wind': ['low', 'mid', 'high']
}

# Calculate Probabilities

""" Start of code from TA """
# Weather does not have any parents so all we need are the marginal probabilities of observing each weather type
weather_marginal = (df['weather'].value_counts()/len(df['weather'])).round(3)
weather_marginal = np.array([[value] for value in weather_marginal])
print(weather_marginal) # Seems right with the info from assignment


# Joint Propabilities
# Create dict where key=parent, value=child
var_dict = {'weather': ['precipitation', 'wind'],
           'precipitation': ['temp_max'],
           'wind': ['temp_min'],
           }


""" # Create conditional distributions and store results in a list
cpd_lst = []
for key, value in var_dict.items():
   length = len(value)
   for i in range(length):
       value_given_key = df.groupby(key)[value[i]].value_counts(normalize=True).sort_index()
       cpd = value_given_key.unstack(fill_value=0).to_numpy().T
       cpd_lst.append(cpd)   """
""" End code from TA """


""" Code from TA gave the wrong shapes of the CPDs """
# Create conditional distributions and store results in a list
cpd_lst = []
for key, value in var_dict.items():
    for child in value:
        # Group by the key and count occurrences of each state in the child, normalized
        grouped = df.groupby(key)[child].value_counts(normalize=True)
        # Unstack and reindex with all possible states to ensure complete shape
        cpd = grouped.unstack(fill_value=0).reindex(columns=states[child], index=states[key], fill_value=0).to_numpy().T
        cpd_lst.append(cpd)
# Print the shapes of the four CPDs
""" for cpd_array in cpd_lst:
    print(cpd_array.shape) """

# Note that we get 3 Nan vslues in the above conditional distributions. This is because one of the type of precipitation (low) did not contain any relation with temp_max.
# Therefore, normalization, does not produce the intended result.
# To mitigate this, we replace Nan with the equal probability within the three values, i.e., 0.33

cpd_lst[2][:,0] = .33
print(cpd_lst)
# Print the shapes of the four CPDs
""" for cpd_array in cpd_lst:
    print(cpd_array) """

# https://pgmpy.org/models/bayesiannetwork.html
# Zoom out and use meny bar to find Exact Inference and Approximate Inference for the next tasks

#cpd_lst

# Creating tabular conditional probability distribution

# CPD for precipitation given weather
weather_cpd = TabularCPD(variable='weather', variable_card=5,
                         values=weather_marginal, # has no evidence
                         state_names={'weather': ['drizzle', 'rain', 'sun', 'snow', 'fog']})

precipitation_cpd = TabularCPD(variable='precipitation', variable_card=3,
                               values=cpd_lst[0],
                               evidence=['weather'], evidence_card=[5],
                               state_names={'precipitation': ['low', 'mid', 'high'],
                                            'weather': ['drizzle', 'rain', 'sun', 'snow', 'fog']})

temp_max_cpd = TabularCPD(variable='temp_max', variable_card=3,
                      values=cpd_lst[2],
                      evidence=['precipitation'], evidence_card=[3],
                      state_names={'temp_max': ['low', 'mid', 'high'],
                                    'precipitation': ['low', 'mid', 'high']})

wind_cpd = TabularCPD(variable='wind', variable_card=3,
                      values=cpd_lst[1],
                      evidence=['weather'], evidence_card=[5],
                      state_names={'wind': ['low', 'mid', 'high'],
                                    'weather': ['drizzle', 'rain', 'sun', 'snow', 'fog']})

temp_min_cpd = TabularCPD(variable='temp_min', variable_card=3,
                      values=cpd_lst[2],
                      evidence=['wind'], evidence_card=[3],
                      state_names={'temp_min': ['low', 'mid', 'high'],
                                    'wind': ['low', 'mid', 'high']})

# Add CPDs and factors to the model

weather_model.add_cpds(weather_cpd, precipitation_cpd, wind_cpd, temp_max_cpd, temp_min_cpd)
print(weather_model)

# Check if model is consistent
print(weather_model.check_model())

# Viewing nodes of the model
weather_model.nodes()

# Viewing edges of the model
weather_model.edges()

# Print the probability table of the weather node
print(weather_cpd)

# Print the probability table of the wind node
print(wind_cpd)

exit()

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


