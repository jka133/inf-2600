import numpy as np
import pandas as pd

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# Import data, make a copy of the original
df0 = pd.read_csv('seattle-weather.csv')
dfc1 = df0.copy()
dfc1.head()

# Get characteristics of dataset including columns with missing data as well:
dfc1.info()

# Checking the unique values in the 'weather' column
unique_fields = dfc1['weather'].unique()
unique_fields = unique_fields.tolist()
print("Unique fields for weather:\n", unique_fields)
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
print('Is missing data')
print(df.isnull().any())

num_stdv = 1

# Define the labels dictionary
labels = {
    'precipitation': ['low', 'mid', 'high'], 
    'temp_max': ['low', 'mid', 'high'], 
    'temp_min': ['low', 'mid', 'high'], 
    'wind': ['low', 'mid', 'high'], 
    'weather': unique_fields
} 

# Calculating mean and standard deviation to use in bounds
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
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
for type in continuous_lst:
    df[type] = df.apply(label_data, args=(type, bounds, labels), axis=1)

print(df.describe())
print(df)

# Define the hierarchy
weather_model = BayesianNetwork([
    ('weather', 'precipitation'),
    ('weather', 'wind'),
    ('precipitation', 'temp_max'),
    ('wind', 'temp_min')
])

# Fitting the cpds from dataframe df using MLE
weather_model.fit(df, estimator=MaximumLikelihoodEstimator, state_names=labels) 
# https://pgmpy.org/_modules/pgmpy/models/BayesianNetwork.html#BayesianNetwork.fit

# Check if model is consistent
print("My weather model is valid:")
print(weather_model.check_model())

# Viewing nodes of the model
weather_model.nodes()

# Viewing edges of the model
weather_model.edges()

# Print the probability table of the weather node
# https://pgmpy.org/models/bayesiannetwork.html#pgmpy.models.BayesianNetwork.BayesianNetwork.get_cpds 
print(weather_model.get_cpds('weather'))
# This seems correct given the info from assignement. weather' - (Rain: 44%; Sunny: 44%; Other (180): 12%).

# Print the probability table of the wind node
print(weather_model.get_cpds('wind'))

# Independcies in the model
# Checking independcies of a particular node
# https://pgmpy.org/base/base.html#pgmpy.base.DAG.local_independencies 
print(f"Independencies in precipitation node: {weather_model.local_independencies('precipitation')}")
print(f"Independencies in temp_max node: {weather_model.local_independencies('temp_max')}")
print(f"Independencies in wind node: {weather_model.local_independencies('wind')}")
print(f"Independencies in temp_min node: {weather_model.local_independencies('temp_min')}")

from pgmpy.inference import VariableElimination

print("\n--- Question 1.2.1 ---\n")
# https://pgmpy.org/exact_infer/ve.html using the query feature
# https://pgmpy.org/exact_infer/ve.html#pgmpy.inference.ExactInference.VariableElimination.query
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
# (a) Calculate all the possible joint probability and determine the best probable condition. Explain your results?
j_prob_a = var_elim.query(variables=['weather', 'precipitation', 'temp_max', 'temp_min', 'wind'])
#j_prob is a huge table oof all the joint probabilities in the network
max_prob_a = j_prob_a.values.max()
max_idx_a = j_prob_a.values.argmax()

# https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
max_idx_a = np.unravel_index(max_idx_a, j_prob_a.values.shape)
print("\nProbability and state for most probable condition")
print(max_prob_a, max_idx_a)

# (b) What is the most probable condition for precipitation, wind and weather, combined?
j_prob_b = var_elim.query(variables=['weather', 'precipitation', 'wind'])
max_prob_b = j_prob_b.values.max()
max_idx_b = j_prob_b.values.argmax()
max_idx_b = np.unravel_index(max_idx_b, j_prob_b.values.shape)
print("\nProbability and state for most probable condition")
print(max_prob_b, max_idx_b)

print("\n--- Question 1.2.3 ---\n")
# Find the probability associated with each weather, given that the precipitation is medium? Explain your result.
j_prob_3 = var_elim.query(variables=['weather'], evidence={'precipitation': 'mid'})
max_prob_3 = j_prob_3.values.max()
max_idx_3 = j_prob_3.values.argmax()
print('\nProbabilities for each weather given mid precipitation')
print(j_prob_3)

print("\n--- Question 1.2.4 ---\n")
# What is the probability of each weather condition given that precipitation is medium and wind is low or medium? 
#Explain your method and results. How does the result change with the addition of wind factor compared to question 3 of Task 1.2?
j_prob_4_l_w = var_elim.query(variables=['weather'], evidence={'precipitation': 'mid', 'wind': 'low'})
j_prob_4_m_w = var_elim.query(variables=['weather'], evidence={'precipitation': 'mid', 'wind': 'mid'})

wind_prob = df['wind'].value_counts().reindex(['low', 'mid'])/df['wind'].value_counts().reindex(['low', 'mid']).sum()
l_w_prior = wind_prob[0]
m_w_prior = wind_prob[1]

print(f'\nPrior probabilities for low and mid wind:\n{l_w_prior, m_w_prior}')
prob_tab = j_prob_4_l_w * l_w_prior + j_prob_4_m_w * m_w_prior
print('\nProbabilities for each weather given mid precipitation and mid or low wind')
print(prob_tab)

# Use prior and max_prob_4 (1&2) and max_idx_4 (1&2) to make the probability
from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling

sample_size = 100000
app_inference = BayesianModelSampling(weather_model)

print("\n--- Question 1.3.1 ---\n")
# (a) of Task 1.2 - What is the probability of high wind when the weather is sunny?
# https://pgmpy.org/approx_infer/bn_sampling.html#pgmpy.sampling.Sampling.BayesianModelSampling.likelihood_weighted_sample 
sample_a = app_inference.likelihood_weighted_sample(evidence=[State('weather','sun')], size=sample_size)
high_wind_samples = sample_a[sample_a['wind'] == 'high']
print('Probability of high wind given sun')
h_wind_given_sun = len(high_wind_samples)/sample_size
print(h_wind_given_sun)

# (b) of Task 1.2 - What is the probability of sunny weather when the wind is high?
sample_b = app_inference.likelihood_weighted_sample(evidence=[State('wind','high')], size=sample_size)
sun_sample = sample_b[sample_b['weather'] == 'sun']
print('\nprobability of sun given high wind')
sun_given_h_wind = len(sun_sample)/sample_size
print(sun_given_h_wind)

print("\n--- Question 1.3.2 ---\n")
# Repeat Q.2 . (a) of Task 1.2 - Calculate all the possible joint probability and determine the best probable condition. Explain your results?
# https://pgmpy.org/approx_infer/bn_sampling.html#pgmpy.sampling.Sampling.BayesianModelSampling.forward_sample
full_sample = app_inference.forward_sample(size=sample_size)
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
occurences_A = full_sample.groupby(['weather', 'wind', 'precipitation', 'temp_max', 'temp_min']).size()
joint_probabilities_a = occurences_A.div(sample_size)

# Find the most probable condition
most_probable_condition_a = joint_probabilities_a.idxmax()
most_probable_probability_a = joint_probabilities_a.max()

print("\nMost Probable Condition and its Probability:")
print(most_probable_condition_a, most_probable_probability_a)

# (b) of Task 1.2 - What is the most probable condition for precipitation, wind and weather, combined?
occurences_b = full_sample.groupby(['weather', 'wind', 'precipitation']).size()
joint_probabilities_b = occurences_b.div(sample_size)

# Find the most probable condition
most_probable_condition_b = joint_probabilities_b.idxmax()
most_probable_probability_b = joint_probabilities_b.max()
print("\nMost Probable Condition and its Probability:")
print(most_probable_condition_b, most_probable_probability_b)

from pgmpy.inference import ApproxInference

print("\n--- Question 1.3.3 ---\n")
# Find the probability associated with each weather, given that the precipitation is medium? Explain your result.
# https://pgmpy.org/approx_infer/approx_infer.html#pgmpy.inference.ApproxInference.ApproxInference 
# https://pgmpy.org/approx_infer/approx_infer.html#pgmpy.inference.ApproxInference.ApproxInference.query
inf = ApproxInference(weather_model)
prob_weather_mid_p = inf.query(variables = ['weather'], n_samples=sample_size, evidence={'precipitation': 'mid'})
print('Probability for each weather given mid precipitation\n')
print(prob_weather_mid_p)

print("\n--- Question 1.3.4 ---\n")
# What is the probability of each weather condition given that precipitation is medium and wind is low or medium?
# Explain your method and results. How does the result change with the addition of wind factor compared to question 3 of Task 1.2?
# https://pgmpy.org/approx_infer/bn_sampling.html#pgmpy.sampling.Sampling.BayesianModelSampling.rejection_sample
l_wind_sample = app_inference.rejection_sample(evidence=[State('precipitation', 'mid'), State('wind', 'low')], size=sample_size)
prob_l = l_wind_sample['weather'].value_counts(normalize=True)

m_wind_sample = app_inference.rejection_sample(evidence=[State('precipitation', 'mid'), State('wind', 'mid')], size=sample_size)
prob_m = m_wind_sample['weather'].value_counts(normalize=True)

print('Probability of weather given mid precipitation and low or mid wind\n')
print(prob_l * l_w_prior + prob_m * m_w_prior)

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

#https://pgmpy.org/_modules/pgmpy/models/BayesianNetwork.html#BayesianNetwork.fit
weather_model2.fit(df, estimator=MaximumLikelihoodEstimator, state_names=labels) 
weather_model3.fit(df, estimator=MaximumLikelihoodEstimator, state_names=labels)

var_elim2 = VariableElimination(weather_model2)
var_elim3 = VariableElimination(weather_model3)

j_prob_2 = var_elim2.query(variables=['weather', 'precipitation', 'temp_max', 'temp_min', 'wind'])
j_prob_3 = var_elim3.query(variables=['weather', 'precipitation', 'temp_max', 'temp_min', 'wind'])

max_prob_2 = j_prob_2.values.max()
max_idx_2 = j_prob_2.values.argmax()
max_idx_2 = np.unravel_index(max_idx_2, j_prob_2.values.shape)
print('Model 2')
print('Maximum joint probability and state giving maximum probability')
print(max_prob_2, max_idx_2)

max_prob_3 = j_prob_3.values.max()
max_idx_3 = j_prob_3.values.argmax()
max_idx_3 = np.unravel_index(max_idx_3, j_prob_3.values.shape)
print('\nModel 3')
print('Maximum joint probability and state giving maximum probability')
print(max_prob_3, max_idx_3)

correlation_matrix = df0[['precipitation', 'temp_max', 'temp_min', 'wind']].corr()
print("Pearson Correlation Coefficient Matrix:")
print(correlation_matrix)