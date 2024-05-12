
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




# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
# The ground work of chosing paramteres (nodes) for the Bayesian Network