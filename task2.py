
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

print(df.corr())

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
    'Precipitation_Type': ['one', 'two', 'three'], 
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
    'Precipitation_Type': [min(unique_fields) + 1, pt_mean, max(unique_fields) - 1], # To ensure three unique values
    'Precipitation_Intensity': [pi_mean, pi_mean + num_stdv * pi_stdv, pi_mean + 2 * num_stdv * pi_stdv] # Heavily left dominant
} 

# Apply the labeling function to the 'precipitation', 'temp_max', 'temp_min', and 'wind' columns
for key, value in labels.items():
    df[key] = df.apply(label_data, args=(key, bounds, labels), axis=1)

""" dfc1 = df.copy()
dfc1.replace('low', 0, inplace=True)
dfc1.replace('mid', 1, inplace=True)
dfc1.replace('high', 2, inplace=True)
print(dfc1.corr()) # Compare with the pearson corrolation from before """

print(df.describe())

# Let's show all columns with missing data as well:
df[df.isnull().any(axis=1)] # any missing data in columns
df.isnull().any()

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
# The ground work of chosing paramteres (nodes) for the Bayesian Network

model = BayesianNetwork([
    ('Air_temp_Act', 'Rel_Humidity_act'),
    ('Air_temp_Act', 'Precipitation_Type'),
    ('Rel_Humidity_act', 'Precipitation_Type'),
    ('Rel_Humidity_act', 'Wind_Speed_avg'),
    ('Wind_Speed_avg', 'Wind_Direction_vct'),
    ('Precipitation_Type', 'Precipitation_Intensity')
])

model.fit(df, estimator=MaximumLikelihoodEstimator, state_names=labels) # Tip from Benjamin Danielsen to fit CPDs
print(f"model is valid: {model.check_model()}")

print("\n--- Question 2.2.1 ---\n")
# Question 2.2.1 (a) What is the prob of precipitation type 'one' given 'mid' actual air temp, (b) and reverse?
var_elim = VariableElimination(model)
phi_query_a = var_elim.query(variables=['Precipitation_Type'], evidence={'Air_temp_Act': 'mid'})
print("\nProbability of precipitation types when the actual air temp is mid:")
print(phi_query_a)


phi_query_b = var_elim.query(variables=['Air_temp_Act'], evidence={'Precipitation_Type': 'one'})
print("\nProbability of actual air temp  when the is precipitation type is 'one':")
print(phi_query_b)

print("\n--- Question 2.2.2 ---\n")
# Question 2:
# (a) Calculate all the possible joint probability and determine the best probable condition. Explain your results?
joint_probability_a = var_elim.query(variables=['Air_temp_Act', 'Rel_Humidity_act', 'Wind_Speed_avg',
                                                'Wind_Direction_vct', 'Precipitation_Type', 'Precipitation_Intensity'])

max_probability_a = joint_probability_a.values.max()
max_index_a = joint_probability_a.values.argmax()
print(max_probability_a, max_index_a)
# Find it

# (b) What is the most probable condition for Air_temp_Act, Precipitation_Type and Rel_Humidity_act, combined?
joint_probability_b = var_elim.query(variables=['Air_temp_Act', 'Precipitation_Type', 'Rel_Humidity_act'])
max_probability_b = joint_probability_b.values.max()
max_index_b = joint_probability_b.values.argmax()
print(max_probability_b, max_index_b)
# Find the variant that yield the highest

print("\n--- Question 2.2.3 ---\n")
# Find the probability associated with each Air_temp_Act, given that the Precipitation_Intensity is medium? Explain your result.
joint_probability_3 = var_elim.query(variables=['Air_temp_Act'], evidence={'Precipitation_Intensity': 'mid'})
max_probability_3 = joint_probability_3.values.max()
max_index_3 = joint_probability_3.values.argmax()
print("Probability of actual air temp  when the is precipitation intensity is 'mid':")
print(joint_probability_3)
print(max_probability_3, max_index_3)

print("\n--- Question 2.2.4 ---\n")
# Question 4. What is the probability of each Air_temp_Act given that Precipitation_Intensity is medium and Wind_Speed_avg is low or medium? 
# Explain your method and results. How does the result change with the addition of Wind_Speed_avg factor compared to 2.2.3?
joint_probability_4_low = var_elim.query(variables=['Air_temp_Act'], evidence={'Precipitation_Intensity': 'mid', 'Wind_Speed_avg': 'low'})
max_probability_4_1 = joint_probability_4_low.values.max()
max_index_4_1 = joint_probability_4_low.values.argmax()
print("Probability of actual air temp  when the is Wind_Speed_avg is 'low':")
print(joint_probability_4_low)
#print(max_probability_4_1, max_index_4_1)

joint_probability_4_mid = var_elim.query(variables=['Air_temp_Act'], evidence={'Precipitation_Intensity': 'mid', 'Wind_Speed_avg': 'mid'})
max_probability_4_2 = joint_probability_4_mid.values.max()
max_index_4_2 = joint_probability_4_mid.values.argmax()
print("Probability of actual air temp  when the is Wind_Speed_avg is 'mid':")
print(joint_probability_4_mid)
#print(max_probability_4_2, max_index_4_2)

wsa_marginal = (df['Wind_Speed_avg'].value_counts().reindex(['low', 'mid', 'high'])/len(df['Wind_Speed_avg'])).round(3)
wsa_marginal = np.array([[value] for value in wsa_marginal])
low_wind_prior,mid_wind_prior = wsa_marginal[0][0], wsa_marginal[1][0]
print(f'Prior probabilities for low and mid Wind_Speed_avg: \n{low_wind_prior, mid_wind_prior}')
# Use the priors and the probabilities in the report to make the probability

### TASK 2.3

app_inference = BayesianModelSampling(model)
sample_size = 100000

print("\n--- Question 2.3.1 ---\n")
# Question 2.2.1 (a) What is the prob of precipitation type 'one' given 'mid' actual air temp, (b) and reverse?
sample_a = app_inference.likelihood_weighted_sample(evidence=[State('Air_temp_Act', 'mid')], size=sample_size)
p_one_samples = sample_a[sample_a['Precipitation_Type'] == 'one']
one_p_given_m_ata = len(p_one_samples)/sample_size
print('Probability of precipitation type one given act air temp mid')
print(one_p_given_m_ata)

sample_b = app_inference.likelihood_weighted_sample(evidence=[State('Precipitation_Type', 'one')], size=sample_size)
ata_mid_samples = sample_b[sample_b['Air_temp_Act'] == 'mid']
m_ata_given_one_p = len(ata_mid_samples)/sample_size
print('Probability of act air temp mid given precipitation type one')
print(m_ata_given_one_p)


print("\n--- Question 2.3.2 ---\n")
# Question 2:
# (a) Calculate all the possible joint probability and determine the best probable condition. Explain your results?
full_sample = app_inference.forward_sample(size=sample_size)
occurences_A = full_sample.groupby(['Air_temp_Act', 'Rel_Humidity_act', 'Wind_Speed_avg',
                                    'Wind_Direction_vct', 'Precipitation_Type', 'Precipitation_Intensity']).size()
joint_probabilities_a = occurences_A.div(sample_size)
# Find the most probable condition
most_probable_condition_a = joint_probabilities_a.idxmax()
most_probable_probability_a = joint_probabilities_a.max()

print("\nMost Probable Condition and its Probability:")
print(most_probable_condition_a, most_probable_probability_a)
# Find it

# (b) What is the most probable condition for Air_temp_Act, Precipitation_Type and Rel_Humidity_act, combined?
occurences_b = full_sample.groupby(['Air_temp_Act', 'Precipitation_Type', 'Rel_Humidity_act']).size()
joint_probabilities_b = occurences_b.div(sample_size)
# Find the most probable condition
most_probable_condition_b = joint_probabilities_b.idxmax()
most_probable_probability_b = joint_probabilities_b.max()

print("\nMost Probable Condition and its Probability:")
print(most_probable_condition_b, most_probable_probability_b)

print("\n--- Question 2.3.3 ---\n")
# Find the probability associated with each Air_temp_Act, given that the Precipitation_Intensity is medium? Explain your result.
inf = ApproxInference(model)
prob_weather_mid_p = inf.query(variables = ['Air_temp_Act'], n_samples=sample_size, evidence={'Precipitation_Intensity': 'mid'})
print(prob_weather_mid_p)

print("\n--- Question 2.3.4 ---\n")
# Question 4. What is the probability of each Air_temp_Act given that Precipitation_Intensity is medium and Wind_Speed_avg is low or medium? 
# Explain your method and results. How does the result change with the addition of Wind_Speed_avg factor compared to 2.3.3?
l_wsa_sample = app_inference.rejection_sample(evidence=[State('Precipitation_Intensity', 'mid'), State('Wind_Speed_avg', 'low')], size=sample_size)
prob_l = l_wsa_sample['Air_temp_Act'].value_counts(normalize=True)
print(prob_l)

m_wsa_sample = app_inference.rejection_sample(evidence=[State('Precipitation_Intensity', 'mid'), State('Wind_Speed_avg', 'mid')], size=sample_size)
prob_m = m_wsa_sample['Air_temp_Act'].value_counts(normalize=True)
print(prob_m)

# Use the prior probabilities:
low_wind_prior, mid_wind_prior
# Do the math in report