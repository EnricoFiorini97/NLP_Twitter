# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Machine Learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition


def main() -> None:
    df = pd.read_csv("dataset.csv", delimiter=',')
    print(df)

if __name__ == "__main__":
    main()