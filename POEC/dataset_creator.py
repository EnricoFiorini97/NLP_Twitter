import sys
import os
import pandas as pd

from pathlib import Path

from pandas.core.indexes import category

def main() -> None:
    columns = ["F_NAME", "F_CONTENT", "CATEGORY"]

    folders = [
        {"Artworks":"/home/enrico/Documents/repos/NLP_Twitter/ML_dataset/artworks"},
        {"Festivities":"/home/enrico/Documents/repos/NLP_Twitter/ML_dataset/festivities"},
        {"History":"/home/enrico/Documents/repos/NLP_Twitter/ML_dataset/history"},
        {"OnThisDay":"/home/enrico/Documents/repos/NLP_Twitter/ML_dataset/OTD"},
        {"Promotions":"/home/enrico/Documents/repos/NLP_Twitter/ML_dataset/promotions"},
        {"VIP_CIT":"/home/enrico/Documents/repos/NLP_Twitter/ML_dataset/vip_cit"}
    ]

    df = pd.DataFrame(columns=columns)
    for folder in folders:
        for file in os.listdir(list(folder.values())[0]):
            content = ""
            try:
                int(os.path.split(f"{list(folder.values())[0]}/{file}")[1].replace(Path(f"{folder.items}/{file}").suffix, ""))
                with open(f"{list(folder.values())[0]}/{file}") as f:
                    content = f.readlines()[0]
            except:
                continue
            f_name = list(folder.keys())[0] + "-" + file
            category = list(folder.keys())[0]
            df.loc[len(df)] = [f_name, content, category]

        df.to_csv("dataset.csv")
            
if __name__ == "__main__":
    main()