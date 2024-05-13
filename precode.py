#pip install pgmpy

# Including the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator

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
p_mean, p_stdv = df['precipitation'].mean(), df['precipitation'].std()

t_max_mean, t_max_stdv = df['temp_max'].mean(), df['temp_max'].std()

t_min_mean, t_min_stdv = df['temp_min'].mean(), df['temp_min'].std()

w_mean, w_stdv = df['wind'].mean(), df['wind'].std()

bounds = {
    'precipitation': [p_mean - num_stdv * p_stdv, p_mean, p_mean + num_stdv * p_stdv],
    'temp_max': [t_max_mean - num_stdv * t_max_stdv, t_max_mean, t_max_mean + num_stdv * t_max_stdv],
    'temp_min': [t_min_mean - num_stdv * t_min_stdv, t_min_mean, t_min_mean + num_stdv * t_min_stdv],
    'wind': [w_mean - num_stdv * w_stdv, w_mean, w_mean + num_stdv * w_stdv]
}

# Change values in columns temp_max, temp_min and wind according to the boundaries. The change should be according to the labels.
def label_data(row, column, bounds, labels):
    if row[column] < bounds[column][0]:
        return labels[column][0]
    if row[column] > bounds[column][2]:
        return labels[column][2]
    return labels[column][1]

# Apply the labeling function to the 'precipitation', 'temp_max', 'temp_min', and 'wind' columns
df['precipitation'] = df.apply(label_data, args=('precipitation', bounds, labels), axis=1)
df['temp_max'] = df.apply(label_data, args=('temp_max', bounds, labels), axis=1)
df['temp_min'] = df.apply(label_data, args=('temp_min', bounds, labels), axis=1)
df['wind'] = df.apply(label_data, args=('wind', bounds, labels), axis=1)

print(df.describe())
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
#weather_marginal = (df['weather'].value_counts()/len(df['weather'])).round(3)
weather_marginal = (df['weather'].value_counts().reindex(['drizzle', 'rain', 'sun', 'snow', 'fog'])/len(df['weather'])).round(3)
# Is this not better?
weather_marginal = np.array([[value] for value in weather_marginal])
print(weather_marginal) # Seems right with the info from assignment

# Joint Propabilities
# Create dict where key=parent, value=child
var_dict = {'weather': ['precipitation', 'wind'],
           'precipitation': ['temp_max'],
           'wind': ['temp_min'],
           }

# Create conditional distributions and store results in a list
cpd_lst = []
for key, value in var_dict.items():
    for child in value:
        # Group by the key and count occurrences of each state in the child, normalized
        grouped = df.groupby(key)[child].value_counts(normalize=True)
        # Unstack and reindex with all possible states to ensure complete shape
        cpd = grouped.unstack(fill_value=0).reindex(columns=states[child], index=states[key], fill_value=0).to_numpy().T
        cpd_lst.append(cpd)
# The loop above is from TA, but fixed but ChatGPT

cpd_lst[2][:,0] = .33
print(cpd_lst)

# Creating tabular conditional probability distribution

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
                      values=cpd_lst[3],
                      evidence=['wind'], evidence_card=[3],
                      state_names={'temp_min': ['low', 'mid', 'high'],
                                    'wind': ['low', 'mid', 'high']})

# Add CPDs and factors to the model
weather_model.add_cpds(weather_cpd, precipitation_cpd, wind_cpd, temp_max_cpd, temp_min_cpd)

# Check if model is consistent
print("My weather model is valid:")
print(weather_model.check_model())

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
print(f"Independencies in precipitation node: {weather_model.local_independencies('precipitation')}")
print(f"Independencies in temp_max node: {weather_model.local_independencies('temp_max')}")
print(f"Independencies in wind node: {weather_model.local_independencies('wind')}")
print(f"Independencies in temp_min node: {weather_model.local_independencies('temp_min')}")

from pgmpy.inference import VariableElimination

print("\n--- Question 1.2.1 ---\n")
# Question 1: 
# https://pgmpy.org/exact_infer/ve.html using the query feature
#(a) What is the probability of high wind when the weather is sunny? 
var_elim = VariableElimination(weather_model)
phi_query_a = var_elim.query(variables=['wind'], evidence={'weather': 'sun'})
print("\nProbability of wind when the weather is sunny:")
print(phi_query_a)

#(b) What is the probability of sunny weather when the wind is high?
phi_query_b = var_elim.query(variables=['weather'], evidence={'wind': 'high'})
print("\nProbability of weather when the wind is high:")
print(phi_query_b)

print("\n--- Question 1.2.2 ---\n")
# Question 2:
# (a) Calculate all the possible joint probability and determine the best probable condition. Explain your results?
j_prob_a = var_elim.query(variables=['weather', 'precipitation', 'temp_max', 'temp_min', 'wind'])
#j_prob is a huge table oof all the joint probabilities in the network
max_prob_a = j_prob_a.values.max()
max_idx_a = j_prob_a.values.argmax()
print(j_prob_a)
print(max_prob_a, max_idx_a)

# (b) What is the most probable condition for precipitation, wind and weather, combined?
j_prob_b = var_elim.query(variables=['weather', 'precipitation', 'wind'])
max_prob_b = np.max(j_prob_b.values)
max_idx_b = np.argmax(j_prob_b.values)
print(j_prob_b)
print(max_prob_b, max_idx_b)
# Find the variant of weather, precip and wind; weather(drizzle) | precipitation(mid)  | wind(mid)

print("\n--- Question 1.2.3 ---\n")
# Question 3. Find the probability associated with each weather, given that the precipitation is medium? Explain your result.
j_prob_3 = var_elim.query(variables=['weather'], evidence={'precipitation': 'mid'})
max_prob_3 = j_prob_3.values.max()
max_idx_3 = j_prob_3.values.argmax()
print(j_prob_3)
print(max_prob_3, max_idx_3)
# Search for it in the table

print("\n--- Question 1.2.4 ---\n")
# Question 4. What is the probability of each weather condition given that precipitation is medium and wind is low or medium? 
#Explain your method and results. How does the result change with the addition of wind factor compared to question 3 of Task 1.2?
j_prob_4_l_w = var_elim.query(variables=['weather'], evidence={'precipitation': 'mid', 'wind': 'low'})
j_prob_4_m_w = var_elim.query(variables=['weather'], evidence={'precipitation': 'mid', 'wind': 'mid'})

wind_prob = df['wind'].value_counts().reindex(['low', 'mid'])/df['wind'].value_counts().reindex(['low', 'mid']).sum()
l_w_prior = wind_prob[0]
m_w_prior = wind_prob[1]

print(f'Prior probabilities for low and mid wind:\n{l_w_prior, m_w_prior}')
prob_tab = j_prob_4_l_w * l_w_prior + j_prob_4_m_w * m_w_prior
print(prob_tab)

# Use prior and max_prob_4 (1&2) and max_idx_4 (1&2) to make the probability

from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling

sample_size = 100000
app_inference = BayesianModelSampling(weather_model)

print("\n--- Question 1.3.1 ---\n")
# Repeat Q.1. (a) of Task 1.2 - What is the probability of high wind when the weather is sunny?
sample_a = app_inference.likelihood_weighted_sample(evidence=[State('weather','sun')], size=sample_size)
high_wind_samples = sample_a[sample_a['wind'] == 'high']
h_wind_given_sun = len(high_wind_samples)/sample_size
print('Probability of high wind given sun')
print(h_wind_given_sun)

# Repeat Q.1. (b) of Task 1.2 - What is the probability of sunny weather when the wind is high?
sample_b = app_inference.likelihood_weighted_sample(evidence=[State('wind','high')], size=sample_size)
sun_sample = sample_b[sample_b['weather'] == 'sun']
print('probability of sun given high wind')
sun_given_h_wind = len(sun_sample)/sample_size
print(sun_given_h_wind)

print("\n--- Question 1.3.2 ---\n")
# Repeat Q.2 . (a) of Task 1.2 - Calculate all the possible joint probability and determine the best probable condition. Explain your results?
full_sample = app_inference.forward_sample(size=sample_size)
occurences_A = full_sample.groupby(['weather', 'wind', 'precipitation', 'temp_max', 'temp_min']).size()
joint_probabilities_a = occurences_A.div(sample_size)

# Find the most probable condition
most_probable_condition_a = joint_probabilities_a.idxmax()
most_probable_probability_a = joint_probabilities_a.max()

print("\nMost Probable Condition and its Probability:")
print(most_probable_condition_a, most_probable_probability_a)

# Repeat Q.2 . (b) of Task 1.2 - What is the most probable condition for precipitation, wind and weather, combined?
occurences_b = full_sample.groupby(['weather', 'wind', 'precipitation']).size()
joint_probabilities_b = occurences_b.div(sample_size)

# Find the most probable condition
most_probable_condition_b = joint_probabilities_b.idxmax()
most_probable_probability_b = joint_probabilities_b.max()
print("\nMost Probable Condition and its Probability:")
print(most_probable_condition_b, most_probable_probability_b)

from pgmpy.inference import ApproxInference

print("\n--- Question 1.3.3 ---\n")
# Repeat Q.3 of Task 1.2 - Find the probability associated with each weather, given that the precipitation is medium? Explain your result.
inf = ApproxInference(weather_model)
prob_weather_mid_p = inf.query(variables = ['weather'], n_samples=sample_size, evidence={'precipitation': 'mid'})
print(prob_weather_mid_p)

print("\n--- Question 1.3.4 ---\n")
# Repeat Q.4 of Task 1.2 - What is the probability of each weather condition given that precipitation is medium and wind is low or medium?
# Explain your method and results. How does the result change with the addition of wind factor compared to question 3 of Task 1.2?
l_wind_sample = app_inference.rejection_sample(evidence=[State('precipitation', 'mid'), State('wind', 'low')], size=sample_size)
prob_l = l_wind_sample['weather'].value_counts(normalize=True)
print(prob_l)

m_wind_sample = app_inference.rejection_sample(evidence=[State('precipitation', 'mid'), State('wind', 'mid')], size=sample_size)
prob_m = m_wind_sample['weather'].value_counts(normalize=True)
print(prob_m)

# Do the # Use the prior probabilities:
l_w_prior, m_w_prior
# Do the math in report in report

print("\n--- Task 1.4 ---\n")
""" New models: """
weather_model2 = BayesianNetwork([
    ('weather', 'precipitation'),
    ('weather', 'wind'),
    ('precipitation', 'temp_max'),
    ('precipitation', 'temp_min'),
    ('wind', 'temp_min'),
    ('wind', 'temp_max')]
)
weather_model3 = BayesianNetwork([
    ('weather', 'wind'),
    ('wind', 'precipitation'),
    ('precipitation', 'temp_max'),
    ('precipitation', 'temp_min')]
)

weather_model2.fit(df, estimator=MaximumLikelihoodEstimator, state_names=states) # Tip from Benjamin Danielsen to fit CPDs
weather_model3.fit(df, estimator=MaximumLikelihoodEstimator, state_names=states)

var_elim2 = VariableElimination(weather_model2)
var_elim3 = VariableElimination(weather_model3)

j_prob_2 = var_elim2.query(variables=['weather', 'precipitation', 'temp_max', 'temp_min', 'wind'])
j_prob_3 = var_elim3.query(variables=['weather', 'precipitation', 'temp_max', 'temp_min', 'wind'])

max_prob_2 = j_prob_2.values.max()
max_idx_2 = j_prob_2.values.argmax()
#print(j_prob_2)
print(max_prob_2, max_idx_2)

max_prob_3 = j_prob_3.values.max()
max_idx_3 = j_prob_3.values.argmax()
#print(j_prob_3)
print(max_prob_3, max_idx_3)
