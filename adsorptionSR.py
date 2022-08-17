import numpy as np
import pandas as pd
from pysr import pysr, best, get_hof

# Dataset
f = open("adsorptionDatasets/Langmuir1918methane.csv", "r")
lines = f.readlines()
# take all rows of csv, remove first line, remove "\n", split into cols by ",", and cast to np array
data = np.array([line.strip().split(",") for line in lines[1:]])

# predict second value from the first
# (grab each column)
X = data[:,0]
y = data[:,1]

runs = 1

for i in range(runs):
    equations = pysr(X, y, niterations=5, progress=False, variable_names=["p"],
        binary_operators=["*", "+", "-", "/"],
        unary_operators=[])

    print(best())
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(get_hof())

