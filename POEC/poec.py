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

import spacy

def main() -> None:
    df = pd.read_csv("../raw_dataset/threads-g2-new-ng.txt", delimiter='\t')
    nlp = spacy.load("en_core_web_sm")

    report = ""
    for _, line in enumerate(df["FULL_TEXT"]):
        line2 = nlp(line)
        for token in line2.ents:
            if token.label_ == 'PERSON':
                report += (line + '\n')
                break

    with open ("tmp.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()