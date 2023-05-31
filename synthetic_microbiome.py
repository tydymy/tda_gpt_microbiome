import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.random import default_rng

rng = default_rng()

def generate_sparse_matrix(n, m, s):
    density = 1 - s
    sparse_matrix = sp.random(n, m, density=density, data_rvs=lambda k: rng.integers(0, 1000000, size=k))
    return sparse_matrix.toarray()

def generate_correlated_column(n, s, correlation_type, variable_of_interest):
    #variable_of_interest = rng.integers(1, 101, n)  # Variable of interest for each column
    noise = rng.normal(0, 1, n)  # normally-distributed noise

    # Generate mask for sparsity
    mask = rng.choice([0, 1], size=n, p=[s, 1-s])

    if correlation_type == "linear":
        column = np.clip(variable_of_interest * 10000 + noise, 0, 1000000).astype(int)  # scale to get larger values
    elif correlation_type == "exponential":
        column = np.clip(np.exp(variable_of_interest / 10) + noise, 0, 1000000).astype(int)  # divide to avoid overflow
    elif correlation_type == "log":
        column = np.clip(np.log(variable_of_interest + 1) * 100000 + noise, 0, 1000000).astype(int)  # scale to get larger values
    elif correlation_type == "threshold":
        column = np.clip((variable_of_interest > np.mean(variable_of_interest)).astype(int) * 1000000 + noise, 0, 1000000).astype(int)
    
    return column * mask

def generate_correlated_matrix(n, m, s, num_correlated, correlation_types, variable_of_interest):
    correlation_labels = rng.choice(correlation_types, num_correlated)
    
    correlated_columns = np.array([generate_correlated_column(n, s, correlation_type, variable_of_interest=variable_of_interest) for correlation_type in correlation_labels])

    sparse_matrix = generate_sparse_matrix(n, m-len(correlation_labels), s)  # generate the rest of the matrix

    final_matrix = np.hstack((sparse_matrix, correlated_columns.T))  # transpose correlated_columns to match shapes
    df = pd.DataFrame(final_matrix)
    
    correlation_labels_df = pd.DataFrame(correlation_labels.reshape(1, -1))
    correlation_labels_df.columns = df.columns[-len(correlation_labels):]
    df = pd.concat([correlation_labels_df, df])
    
    return df

# Usage
n = 300  # number of rows
m = 1000    # number of columns
s = 0.7   # sparsity
num_correlated = 10  # number of correlated columns
correlation_types = ["linear", "exponential", "log", "threshold"]  # allowed correlation types
variable_of_interest = rng.integers(1, 101, n)
df = generate_correlated_matrix(n, m, s, num_correlated, correlation_types, variable_of_interest)
